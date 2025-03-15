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

class DesignLogger:
    """Records and keeps the history of designs and refwards."""

    def __init__(self, maxlen,cut_off_list):
        self.maxlen = maxlen
        self.designs = {}
        self.rewards = {}
        self.count = {}
        self.data = {}
        for i in cut_off_list:
            self.designs[i] = deque([], maxlen=maxlen)
            self.rewards[i] = deque([], maxlen=maxlen)
            self.count[i] = 0
            self.data[i] = []
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
        if self.count[body_length] % 3 == 0:  
            # Check the type of design.
            if isinstance(design, dict):
                # Convert dictionary into a string: "key1=value1, key2=value2, ..."
                design_str = ", ".join(str(value) for value in design.values())
            elif isinstance(design, (list, tuple)):
                design_str = str([int(x) for x in design])
            else:
                design_str = str(int(design))
            self.data[body_length].append([self.count[body_length], design_str, float(reward)])

        if self.count[body_length] % 3 == 0:
            table = wandb.Table(data=self.data[body_length], columns=self.columns)
            wandb.log({f"designs_{body_length}": table, "train/design_count": self.count[body_length]})
    def state_dict(self):
        return {
            'designs': list(self.designs),
            'rewards': list(self.rewards),
            'count': self.count
        }

    def load_state_dict(self, state_dict):
        self.designs = deque(state_dict['designs'], maxlen=self.maxlen)
        self.rewards = deque(state_dict['rewards'], maxlen=self.maxlen)
        self.count = state_dict['count']

    def get_designs_and_rewards(self, n,cut_off_length):
        return list(self.designs[cut_off_length])[-n:], list(self.rewards[cut_off_length])[-n:]

    def get_design_count(self):
        return self.count


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
                 cut_off_list = [5,10,15,20]):
        super().__init__(venv)

        self.steps_per_EP= n_steps
        self.n_env = n_env
        self.steps_per_design = int(self.steps_per_EP/self.n_env) # Number of evironment steps each design takes
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
        for i in self.cut_off_list:
            self.last_design_update[i] = 0
            # space_idx = list(self.design_space.keys())[i]
            self.design_optimizer[i] = RobotDesignOptimizer(self.design_space[i],
                                                            self.ent_decay_start, 
                                                            self.ent_decay_end)
        self.ckptr = Checkpointer(os.path.join(logdir, 'ckpts_design_params'))
        self.configure_design_manager(self.steps_per_design,self.batch_size)

    # def step_wait(self):
    #     return self.venv.step_wait()

    def set_design_dist(self, design_dist):
        self.design_dist = design_dist
        
    def get_design_count(self):
        return self.logger.get_design_count()

    def configure_design_manager(self, steps_per_design, maxlen):
        self.steps_per_design = steps_per_design
        self.logger = DesignLogger(maxlen = maxlen,
                                   cut_off_list=self.cut_off_list)

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
        if len(self.designs) < self.n_env:
            for _ in range(self.n_env - len(self.designs)):
                sample = self._sample_design(optimizer = optimizer)  # sample is expected to be a dict.
                sample = {'body_length': cut_off_length, **sample}
                self.designs.append(sample)
        # self.designs = [self._sample_design() for _ in range(self.num_envs)]
        print("designs: ", self.designs)
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
        for i in self.cut_off_list:
            designs_since_update = design_count[i] - self.last_design_update[i]
            if designs_since_update >= self.update_period:
                designs, rewards = self.get_designs_and_rewards(self.batch_size,i)
                print("DESIGNS USED FOR UPDATE", designs)
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
            for design, reward in zip(self.designs, self.rewards):
                self.logger.log(design, reward)
            self.trigger_dist_update()
            self.init_scene()
            dones[:] = True  # Force reset after a design interval
        return obs, rewards, dones, infos
    def state_dict(self):
        return {'designs': self.logger.state_dict()}

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