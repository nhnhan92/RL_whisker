"""Joint optimization of design and control."""

import numpy as np
import torch
import os
import gym
from dl import nest
from dl.ckptr import Checkpointer
from coopt.discrete_robot_optimizer import DiscreteDesignOptimizer


class CoOpt():
    def __init__(self,
                 env,
                 design_space,
                 logdir,
                 steps_per_design,  # Number of evironment steps each design takes
                 batch_size,        # number of designs used in each update
                 update_period,     # number of designs sampled between updates
                 ent_decay_start, # start of entropy decay
                 ent_decay_end   # end of entropy decay
                 ):
        self.logdir = logdir
        self.steps_per_design = steps_per_design
        self.batch_size = batch_size
        self.update_period = update_period
        self.env = env
        self.env.configure_design_manager(
            self.steps_per_design,
            self.batch_size
        )

        self.design_optimizer = DiscreteDesignOptimizer(design_space, ent_decay_start, ent_decay_end)
        self.ckptr = Checkpointer(os.path.join(logdir, 'ckpts_design_params'))
        self.t = 0
        self.design = None
        self.last_design_update = 0
        self._prev_design_grad = None
        self._prev_design_count = 0
        self._ob = None

    def step(self):
        # Log the current step
        if self.t == 0:
            self.design_optimizer.log(self.t)

        # Step env until update.
        if self._ob is None:
            self._ob = self.env.reset()

        for _ in range(self.update_period):
            actions = [self.env.action_space.sample() for _ in range(self.env.num_envs)]
            obs, rewards, dones, info = self.env.step(actions)
            self._ob = obs


        # Update the design distribution
        dt = self.env.num_envs * self.update_period
        self.t += dt
        design_count = self.env.get_design_count()
        designs_since_update = design_count - self.last_design_update
        if designs_since_update >= self.update_period:
            designs, rewards = self.env.get_designs_and_rewards(self.batch_size)
            print("====================================")
            print("designs: ", designs)
            designs = nest.map_structure(torch.from_numpy, designs)
            self.design_optimizer.update(designs, torch.from_numpy(rewards),
                                         self.t)
            self.design_optimizer.log(self.t)
            self.last_design_update = design_count
    
        # Set the design distribution in the environment
        self.env.set_design_dist(self.design_optimizer.get_design_dist())
        return self.t


    def evaluate(self):
        pass

    def save(self):
        state = self.design_optimizer.state_dict()
        self.ckptr.save(state, self.t)

    def load(self, t=None):
        state_dict = self.ckptr.load(t)
        if state_dict is not None:
            self.design_optimizer.load_state_dict(state_dict)
        design_count = self.env.get_design_count()
        self.last_design_update = design_count - (design_count % self.update_period)

    def close(self):
        self.env.close()