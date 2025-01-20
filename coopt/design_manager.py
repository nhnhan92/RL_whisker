import gym.spaces
import numpy as np
from collections import deque
from dl import nest
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
import torch
import wandb
import gym
import torch
import os
from dl.ckptr import Checkpointer
from coopt.discrete_robot_optimizer import DiscreteDesignOptimizer

class DesignLogger:
    """Records and keeps the history of designs and refwards."""

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.designs = deque([], maxlen=maxlen)
        self.rewards = deque([], maxlen=maxlen)
        self.count = 0
        self.columns = ['count', 'design', 'reward']
        self.data = []
        wandb.define_metric("train/design_count", step_metric="train/step")
        wandb.define_metric("designs", step_metric="train/design_count")

    def log(self, design, reward):
        self.designs.append(design)
        self.rewards.append(reward)
        if self.count % 100 == 0:
            if isinstance(design, (list, tuple)):
                design_str = str([int(x) for x in design])
            else:
                design_str = str(int(design))
            self.data.append([self.count, design_str, float(reward)])
        if self.count % 1000 == 0:
            table = wandb.Table(data=self.data, columns=self.columns)
            wandb.log({"designs": table, "train/design_count": self.count})
        self.count += 1

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

    def get_designs_and_rewards(self, n):
        return list(self.designs)[-n:], np.array(self.rewards)[-n:]

    def get_design_count(self):
        return self.count


class DesignManager(VecEnvWrapper):
    """Environment wrapper to handle design sampling/logging.

    This wrapper:
         - samples and sets design parameters at fixed intervals.
         - records and logs the performance of each design.
    """
    
    def __init__(self, venv, design_space=None,n_steps = 100,n_env = 1,update_period = 10):
        super().__init__(venv)
        self.steps_per_EP= n_steps
        self.n_env = n_env
        self.steps_per_design = int(self.steps_per_EP/self.n_env) # Number of evironment steps each design takes
        self.design_count = 0
        self.batch_size = 64 # number of designs used in each update
        self.last_design_update = 0
        self.steps = 0
        self.t = 0
        self.update_period = update_period
        self.design_dist = None
        space = gym.spaces.Discrete(n=9)
        self.design_space = space
        self.designs = None
        self.rewards = np.zeros(self.num_envs)
        ent_decay_start = 10 # start of entropy decay
        ent_decay_end = 1000  # end of entropy decay
        logdir = './test_coopt'
        self.design_optimizer = DiscreteDesignOptimizer(self.design_space, ent_decay_start, ent_decay_end)
        self.ckptr = Checkpointer(os.path.join(logdir, 'ckpts_design_params'))
        self.configure_design_manager(self.steps_per_design,self.batch_size)

    def step_wait(self):
        return self.venv.step_wait()

    def set_design_dist(self, design_dist):
        self.design_dist = design_dist
        
    def get_design_count(self):
        return self.logger.get_design_count()

    def configure_design_manager(self, steps_per_design, maxlen):
        self.steps_per_design = steps_per_design
        self.logger = DesignLogger(maxlen)

    def get_designs_and_rewards(self, n):
        return self.logger.get_designs_and_rewards(n)

    def _unnorm(self, p):
        # space = self.observation_space['design']
        # if isinstance(space, gym.spaces.Box):
        #     return 0.5 * (p.clamp_(-1., 1.) + 1.) * (space.high - space.low) + space.low
        # else:
        #     return p
        return p
    
    def _sample_design(self):
        if self.design_dist is None:
            return np.array(self.design_space.sample())
        with torch.no_grad():
            return self._unnorm(nest.map_structure(
                            lambda x: x.numpy(), self.design_dist.sample()))

    def _sample_mode(self):
        if self.design_dist is None:
            return np.array(self.design_space.sample())
        with torch.no_grad():
            return self._unnorm(nest.map_structure(
                            lambda x: x.numpy(), self.design_dist.mode()))

    def init_scene_with_mode(self):
        self.designs = [self._sample_mode() for _ in range(self.num_envs)]
        self.venv.set_designs(self.designs)
        self.rewards = np.zeros(self.num_envs)

    def init_scene(self):
        self.designs = [self._sample_design() for _ in range(self.num_envs)]
        print("designs: ", self.designs)
        self.venv.set_designs(self.designs)
        self.rewards = np.zeros(self.num_envs)

    def reset(self):
        if self.designs is None:
            print("self.designs is None")
            self.init_scene()
        return self.venv.reset()

    def trigger_dist_update(self):
        # Update the design distribution
        dt = self.n_env * self.update_period
        self.t += dt
        design_count = self.get_design_count()
        print("Design Count =", design_count)
        designs_since_update = design_count - self.last_design_update
        if designs_since_update >= self.update_period:
            designs, rewards = self.get_designs_and_rewards(self.batch_size)
            designs = nest.map_structure(torch.from_numpy, designs)
            self.design_optimizer.update(designs, torch.from_numpy(rewards),
                                         self.t)
            self.design_optimizer.log(self.t)
            self.last_design_update = design_count
    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        # Log the current step
        if self.t == 0:
            self.design_optimizer.log(self.t)

        obs, rewards, dones, infos = self.venv.step_wait()
        dt = self.n_env * self.update_period
        self.t += dt
        self.steps += 1
        self.rewards += rewards
        if self.steps_per_design and self.steps % self.steps_per_design == 0:
            for design, reward in zip(self.designs, self.rewards):
                if self.logger:
                    self.logger.log(design, reward)
            self.init_scene()
            print("FINISHING EPI NOW MOVE TO NEW EPI")
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