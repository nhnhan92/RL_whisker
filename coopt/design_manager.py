import gym.spaces
import numpy as np
from collections import deque
from dl import nest
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
import torch
import wandb
import gym
import os
from dl.ckptr import Checkpointer
from coopt.discrete_robot_optimizer import DiscreteDesignOptimizer
from coopt.robot_optimizer import RobotDesignOptimizer
from itertools import product

from pympler import asizeof

def bytes_to_mb(x):
    return x / (1024 ** 2)

def dump_self_data_size(data, tag=""):
    size_b = asizeof.asizeof(data)
    print(f"[mem]{tag} self.data = {bytes_to_mb(size_b):.2f} MB")



class DesignLogger:
    """Records and keeps the history of designs and refwards."""

    def __init__(self, maxlen,cut_off_list,discrete_combinations):
        self.maxlen = maxlen
        self.discrete_combinations = discrete_combinations
        self.designs = {}
        self.rewards = {}
        self.count = {}
        self.data = {}
        self.average_discrete_design = {}
        for i in cut_off_list:
            self.designs[i] = deque([], maxlen=maxlen)
            self.rewards[i] = deque([], maxlen=maxlen)
            self.count[i] = 0
            self.data[i] = []
            self.average_discrete_design[i] = {}
            for j in range(len(self.discrete_combinations)):
                self.average_discrete_design[i][j] = 0
            wandb.define_metric(f"designs_{i}", step_metric="train/design_count")

        wandb.define_metric("train/design_count", step_metric="train/step")
        self.columns = ['count', 'design', 'reward']
        

    def log(self, design, reward):
        body_length = design['body_length']
        if body_length in self.designs:
            self.designs[body_length].append(design)
            self.rewards[body_length].append(reward)
            self.count[body_length] += 1
        else:
            raise ValueError(f"body_length value {body_length} not found in {self.designs}")
        
        d_val = (design['no_chamber'], design['chamber_length'], design['thickness'])
        idx = self.discrete_combinations.index(d_val)
        if self.average_discrete_design[body_length][idx] != 0:
            if self.average_discrete_design[body_length][idx] < reward:
                self.average_discrete_design[body_length][idx] = reward
        else:
            self.average_discrete_design[body_length][idx] = reward
        
        if self.count[body_length] % 100 == 0:  
            # Check the type of design.
            if isinstance(design, dict):
                # Convert dictionary into a string: "key1=value1, key2=value2, ..."
                design_str = ", ".join(str(value) for value in design.values())
            elif isinstance(design, (list, tuple)):
                design_str = str([int(x) for x in design])
            else:
                design_str = str(int(design))
            self.data[body_length].append([self.count[body_length], design_str, float(reward)])
        if self.count[body_length] % 1000 == 0:
            # dump_self_data_size(self.data, tag=f" #{self.count[body_length]}")
            table = wandb.Table(data=self.data[body_length], columns=self.columns)
            wandb.log({f"designs_{body_length}": table, "train/design_count": self.count[body_length]})


    def state_dict(self):
        sums = {}
        for body_length, inner in self.average_discrete_design.items():
            for key, value in inner.items():
                # Here, key is the "second item" (e.g., '1', '2', '3').
                sums[key] = sums.get(key, 0) + value
        max_sum = max(sums.values())
        min_sum = min(sums.values())
        range = max_sum - min_sum
        extra_rewards = {k: (v - min_sum) / range for k, v in sums.items()}
        return {
            'designs': list(self.designs),
            'rewards': list(self.rewards),
            'count': self.count,
            'extra_rewards': extra_rewards
        }

    def load_state_dict(self, state_dict):
        self.designs = deque(state_dict['designs'], maxlen=self.maxlen)
        self.rewards = deque(state_dict['rewards'], maxlen=self.maxlen)
        self.count = state_dict['count']

    def get_designs_and_rewards(self, n,cut_off_length):
        return list(self.designs[cut_off_length])[-n:], list(self.rewards[cut_off_length])[-n:]

    def get_design_count(self):
        return self.count
# --- LinearSchedule Class (as before) ---
class LinearSchedule():
    def __init__(self, start_val, end_val, start_step, end_step):
        self.sv = start_val
        self.ev = end_val
        self.ss = start_step
        self.es = end_step

    def __call__(self, t):
        if t <= self.ss:
            return self.sv
        if t >= self.es:
            return self.ev
        frac = (t - self.ss) / (self.es - self.ss)
        return self.ev * frac + self.sv * (1 - frac)

class DesignManager(VecEnvWrapper):
    """Environment wrapper to handle design sampling/logging.

    This wrapper:
         - samples and sets design parameters at fixed intervals.
         - records and logs the performance of each design.
    """
    
    def __init__(self, venv, design_space=None,
                 n_steps = 100,
                 n_env = 1,
                 batch_size = 0,
                 update_period = 10,
                 ent_decay_start = 1,
                 ent_decay_end = 10,
                 cut_off_list = [5,10,15,20],
                 test_mode=False,
                 shared_distribution = None):
        super().__init__(venv)
        self.test_mode = test_mode
        # self.steps_per_EP= n_steps
        self.n_env = n_env
        self.steps_per_design = n_steps # Number of evironment steps each design takes
        self.design_count = 0
        self.batch_size = batch_size # number of designs used in each update
        self.steps = 0
        self.t = 0
        self.update_period = update_period  #number of sampled designs that reach before dist update
        self.cut_off_list = cut_off_list
        self.design_space = design_space
        self.designs = None
        self.rewards = np.zeros(self.n_env)
        self.ent_decay_start = ent_decay_start # start of entropy decay
        self.ent_decay_end = ent_decay_end  # end of entropy decay
        logdir = './test_coopt'
        self.design_optimizer = {}
        self.last_design_update = {}
        self.design_avg_extra_reward = LinearSchedule(0, 1, ent_decay_start, ent_decay_end)
        self.discrete_combinations = list(product(self.design_space[100]['no_chamber'], 
                                                  self.design_space[100]['chamber_length'],
                                                  self.design_space[100]['thickness']))
        if not self.test_mode:
            for i in self.cut_off_list:
                self.last_design_update[i] = 0
                # space_idx = list(self.design_space.keys())[i]
                self.design_optimizer[i] = RobotDesignOptimizer(i,
                                                                self.design_space[i],
                                                                self.ent_decay_start, 
                                                                self.ent_decay_end)
        else:
            self.design_optimizer = shared_distribution
        self.ckptr = Checkpointer(os.path.join(logdir, 'ckpts_design_params'))
        self.configure_design_manager(self.batch_size)

    def distribution_state_reference(self):
        shared_dist = []
        for cut_off_length, optimizer in self.design_optimizer.items():
            shared_dist[cut_off_length] = optimizer
        return shared_dist

    def set_design_dist(self, design_dist):
        self.design_dist = design_dist
        
    def get_design_count(self):
        return self.logger.get_design_count()

    def configure_design_manager(self, maxlen):
        self.logger = DesignLogger(maxlen = maxlen,
                                   cut_off_list=self.cut_off_list,
                                   discrete_combinations = self.discrete_combinations)

    def get_designs_and_rewards(self, n,cut_off_length):
        return self.logger.get_designs_and_rewards(n,cut_off_length)

    def _unnorm(self, p):
        # space = self.observation_space['design']
        # if isinstance(space, gym.spaces.Box):
        #     return 0.5 * (p.clamp_(-1., 1.) + 1.) * (space.high - space.low) + space.low
        # else:
        #     return p
        return p
    
    def _sample_design(self,optimizer):
        with torch.no_grad():
            return self._unnorm(nest.map_structure(
                            lambda x: x.numpy().item(), optimizer.sample()))

    def _sample_mode(self,optimizer):
        with torch.no_grad():
            return self._unnorm(nest.map_structure(
                            lambda x: x.numpy(), optimizer.mode()))

    def init_scene_with_mode(self):
        self.designs = [self._sample_mode() for _ in range(self.n_env)]
        self.venv.set_designs(self.designs)
        self.rewards = np.zeros(self.n_env)

    def init_scene(self):
        self.designs = []
        n_design_per_length = self.n_env // len(self.cut_off_list)
        for cut_off_length, optimizer in self.design_optimizer.items():
            for _ in range(n_design_per_length):
                sample = self._sample_design(optimizer = optimizer)  # sample is expected to be a dict.
                sample = {'body_length': cut_off_length, **sample}
                self.designs.append(sample)
        for cut_off_length, optimizer in self.design_optimizer.items():
            if len(self.designs) < self.n_env:
                sample = self._sample_design(optimizer = optimizer)  # sample is expected to be a dict.
                sample = {'body_length': cut_off_length, **sample}
                self.designs.append(sample)
        # self.designs = [self._sample_design() for _ in range(self.num_envs)]
        # print("designs: ", self.designs)
        self.venv.set_designs(self.designs)
        self.rewards = np.zeros(self.n_env)
    
    def reset(self):
        if self.designs is None:
            print("self.designs is None")
            self.init_scene()
        return self.venv.reset()

    def trigger_dist_update(self):
        # Update the design distribution
        design_count = self.get_design_count()
        print("Design Count =", design_count)
        extra_reward_rate = np.zeros(len(self.discrete_combinations))
        state_dict = self.state_dict()
        extra_rewards = state_dict['extra_rewards']
        # print(extra_rewards)
        base_extra_reward = self.design_avg_extra_reward(self.t)
        # print(f'base_extra_reward = {base_extra_reward}')
        for i in self.cut_off_list:
            designs_since_update = design_count[i] - self.last_design_update[i]
            if designs_since_update >= self.update_period:
                designs, rewards = self.get_designs_and_rewards(self.batch_size,i)
                for j, (sample,reward) in enumerate(zip(designs, rewards)):
                    d_val = (sample['no_chamber'], sample['chamber_length'], sample['thickness'])
                    idx = self.discrete_combinations.index(d_val)
                    rewards[j] += base_extra_reward * extra_rewards[idx]
                    # print(f'extra_rewards[{idx}] = {base_extra_reward * extra_rewards[idx]}')
                self.design_optimizer[i].update(designs, rewards,
                                            self.t)
                self.design_optimizer[i].log(self.t)
                self.last_design_update[i] = design_count[i]

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        # Log the current step
        if self.t == 0:
            for i in self.cut_off_list:
                self.design_optimizer[i].log(self.t)

        obs, rewards, dones, infos = self.venv.step_wait()
        dt = self.n_env
        self.t += dt
        self.steps += 1
        self.rewards += rewards
        if self.steps_per_design and self.steps % self.steps_per_design == 0:
            if not self.test_mode:
                for design, reward in zip(self.designs, self.rewards):
                    self.logger.log(design, reward)
                self.trigger_dist_update()
            self.init_scene()
            dones[:] = True  # Force reset after a design interval
        return obs, rewards, dones, infos
    def state_dict(self):
        return self.logger.state_dict()

    def load_state_dict(self, state_dict):
        self.logger.load_state_dict(state_dict['designs'])

    def save(self):
        state = self.design_optimizer.state_dict()
        self.ckptr.save(state, self.t)

    def load(self, t=None):
        state_dict = self.ckptr.load(t)
        if state_dict is not None:
            self.design_optimizer.load_state_dict(state_dict)
        design_count = self.get_design_count()
        self.last_design_update = design_count - (design_count % self.update_period)