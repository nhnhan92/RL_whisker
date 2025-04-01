# -*- coding: utf-8 -*-


__authors__ = ("emenager", "ekhairallah")
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Oct 7 2020"

import os
import numpy as np
import math as m
from typing import Optional
from sofagym.AbstractEnv import AbstractEnv
from sofagym.ServerEnv import ServerEnv
from sofagym.rpc_server import start_scene
from .distribution import update_gibbs_distribution_for_categories
from gym import spaces
import sys
import pathlib
from .design_space.design_space import whiskerdesignspace
import json
import torch
# sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute()))

path = os.path.dirname(os.path.abspath(__file__))
# env_path = os.path.join(path,"sofagym/envs/Whisker/")
class WhiskerEnv:
    # Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    dim_state = 8
    DEFAULT_CONFIG = {"scene": "Whisker",
                      "deterministic": True,
                      "source": [-220, -20, 30],
                      "target": [100, -100, -50],
                      "goalList": [[0, 0, 0]],
                      "goal": False,
                      "start_node": None,
                      "scale_factor": 10,  # equivalent to simulation duration = scale_factor * dt - dt
                      "timer_limit": 60,
                      "timeout": 50,
                      "display_size": (800, 600),
                      "render": 0,
                      "save_data": False,
                      "save_image": False,
                      "save_path": path + "/Results" + "/Whisker",
                      "planning": False,
                      "discrete": False, # True when state space and action space both are discrete
                      "seed": None,
                      "start_from_history": [],  # this number represents the action of the RL environment not steps in SOFA simulation
                      "python_version": "python3",
                      "time_before_start": 0,
                      "dt": 0.01,
                      "design_params": [100,1,20,2.0,0.1,0.0],
                      "nb_actions": -1,
                      "dim_state": dim_state,
                      "init_states": [0,0,100,1,20,2,0.1,0.1],
                      "randomize_states": True,
                      "use_server": False,
                      "zFar":4000
                      }
    def __init__(self, config = None, root=None, use_server: Optional[bool]=None):
        if use_server is not None:
            self.DEFAULT_CONFIG.update({'use_server': use_server})
        self.use_server = self.DEFAULT_CONFIG["use_server"]
        self.env = ServerEnv(self.DEFAULT_CONFIG, config, root=root) if self.use_server else AbstractEnv(self.DEFAULT_CONFIG, config, root=root)
        self.initialize_states()
        if self.env.config["goal"]:
            self.init_goal()
        low = -1
        high = 1
        self.env.action_space = spaces.Box(low=low, high=high, shape=(1,), dtype='float32')
        self.nb_actions = str(self.env.nb_actions)
        low_coordinates = np.array([-1]*self.env.dim_state)
        high_coordinates = np.array([1]*self.env.dim_state)
        self.env.observation_space = spaces.Box(low_coordinates, high_coordinates,
                                            dtype='float32')
        self.body_length_categories = np.array([80,90,100])
    
    # called when an attribute is not found:
    def __getattr__(self, name):
        # assume it is implemented by self.instance
        return self.env.__getattribute__(name)
    
    def initialize_states(self):
        if self.env.config["randomize_states"]:
            self.init_states = self.randomize_init_states()
            # self.env.config.update({'init_states': list(self.init_states)})
            self.env.config["init_states"] = self.init_states
        else:
            self.init_states = self.env.config["init_states"]
        
    def randomize_init_states(self):
        """Randomize initial inclined angle of whisker body

        Returns:
        -------
            init_states: list
                List of random initial states for the environment.
        
        Note:
        ----
            This method should be implemented according to needed random initialization.
        """
        init_body_angle = self.env.np_random.uniform(low=-2*m.pi/180, high=2*m.pi/180, size=None)
        init_states = [init_body_angle,0,100,1,20,2,0,0]
        return init_states
    
    
    def design_changer(self,design_param):
        body_length = design_param['body_length']
        no_chamber = design_param['no_chamber']
        chamber_length = design_param['chamber_length']
        thickness = design_param['thickness']
        pressure1= design_param['pressure1']
        pressure2= design_param['pressure2']

        self.config['design_params'] = [body_length,
                                        no_chamber,
                                        chamber_length,
                                        thickness,
                                       pressure1,pressure2]
        

    def reset(self):
        """Reset simulation.
        """
        # self.initialize_states()

        if self.env.config["goal"]:
            self.init_goal()
        self.env.reset()
        if self.use_server:
            obs = start_scene(self.env.config, self.nb_actions)
            state = np.array(obs['observation'], dtype=np.float32)
        else:
            state = np.array(self.env._getState(self.env.root), dtype=np.float32)
        return state
    

    def get_available_actions(self):
        """Gives the actions available in the environment.

        Parameters:
        ----------
            None.

        Returns:
        -------
            list of the action available in the environment.
        """
        return list(range(int(self.nb_actions)))