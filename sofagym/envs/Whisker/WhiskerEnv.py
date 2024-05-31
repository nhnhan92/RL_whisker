# -*- coding: utf-8 -*-


__authors__ = ("emenager", "ekhairallah")
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Oct 7 2020"

import os
import numpy as np
import torch
from sofagym.AbstractEnv import AbstractEnv
from sofagym.rpc_server import start_scene
from .distribution import update_gibbs_distribution_for_categories
from gym import spaces
import sys
import pathlib
from .design_space.design_space import whiskerdesignspace
import json
# sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute()))
class WhiskerEnv(AbstractEnv):
    """Sub-class of AbstractEnv, dedicated to the gripper scene.

    See the class AbstractEnv for arguments and methods.
    """
    # Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "Whisker",
                      "deterministic": True,
                      "source": [0, 180, 120],
                      "target": [0, -80, 0],
                      "goalList": [[0, 0, 0]],
                      "body": 100,
                      "no_chamber": 2,
                      "start_node": None,
                      "scale_factor": 5,  # equivalent to simulation duration = scale_factor * dt - dt
                      "timer_limit": 250,
                      "timeout": 50,
                      "display_size": (1200, 800),
                      "render": 0,
                      "save_data": False,
                      "save_image": False,
                      "save_path": path + "/Results" + "/Whisker",
                      "planning": False,
                      "discrete": False, # True when state space and action space both are discrete
                      "seed": None,
                      "start_from_history": [],  # this number represents the action of the RL environment not steps in SOFA simulation
                      "python_version": "python3",
                      "dt": 0.01
                      }

    
    def __init__(self, config=None):
        super().__init__(config)
        nb_actions = -1
        low = -1
        high = 1
        self.action_space = spaces.Box(low=low, high=high, shape=(1,), dtype='float32')
        self.nb_actions = str(nb_actions)

        dim_state = 4
        low_coordinates = np.array([-1]*dim_state)
        high_coordinates = np.array([1]*dim_state)
        self.observation_space = spaces.Box(low_coordinates, high_coordinates,
                                            dtype='float32')
        
        ins = whiskerdesignspace()
        self.design_space = ins.design_space()
    

    def step(self, action):
        return super().step(action)
    
    def design_changer(self,design_params):
        self.config['body'] = design_params[0]
        self.config['no_chamber'] = design_params[1]
        
    def reset(self):
        """Reset simulation.

        Note:
        ----
            We launch a client to create the scene. The scene of the program is
            client_<scene>Env.py.

        """
        super().reset()
        self.config.update({'goalPos': self.goal})
        sample = self.sampling_design()
        self.save_json(self.design_space[sample])
        print("SAMPLE = ", self.design_space[sample])
        self.design_changer(self.design_space[sample])
        obs = start_scene(self.config, self.nb_actions)

        return (np.array(obs['observation']))
    
    def sampling_design(self):
        
        initial_logits = torch.ones(len(self.design_space),dtype=torch.float)
        initial_beta = 1.0  # Initial inverse temperature
        self.sample = update_gibbs_distribution_for_categories(self.design_space, initial_logits, initial_beta)
        return self.sample.item()

    def save_json(self,data):
        # Define the file name
        file_name = 'data.json'
        # Define the new data to be added
        write_data = {
            "body_length": data[0],
            "no_chamber": data[1],
            "pressure_1": data[2],
            "pressure_2": data[3],
            "pressure_3": data[4]
        }
        with open(file_name, 'w') as file:
            json.dump(write_data, file, indent=4)

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
