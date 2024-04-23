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

from gym import spaces

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

        dim_state = 3
        low_coordinates = np.array([-1]*dim_state)
        high_coordinates = np.array([1]*dim_state)
        self.observation_space = spaces.Box(low_coordinates, high_coordinates,
                                            dtype='float32')
        
    

    def step(self, action):
        return super().step(action)
    
    def design_changer(self,body_length):
        self.config['body'] = body_length
        pass
    def reset(self):
        """Reset simulation.

        Note:
        ----
            We launch a client to create the scene. The scene of the program is
            client_<scene>Env.py.

        """
        super().reset()
        categories = np.array([80,90,100]) # Categories from 1 to 10
        initial_logits = torch.ones(3,dtype=torch.float)
        initial_beta = 1.0  # Initial inverse temperature
        
        self.config.update({'goalPos': self.goal})
        sample = update_gibbs_distribution_for_categories(categories, initial_logits, initial_beta,update_beta=0)
        print("SAMPLE = ", categories[sample.item()])
        self.design_changer(categories[sample.item()])
        obs = start_scene(self.config, self.nb_actions)
        return (np.array(obs['observation']))

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

class CustomCategoricalDistribution:
    def __init__(self, logits):
        self.dist = torch.distributions.Categorical(logits=logits)

    def sample(self):
        return self.dist.sample()
    
    
def gibbs_distribution_for_categories(categories, logits, beta):
    custom_dist = CustomCategoricalDistribution(logits * beta)
    return custom_dist


# Function to update Gibbs distribution for categories over time and sample
def update_gibbs_distribution_for_categories(categories, initial_logits, initial_beta, update_beta):
    updated_logits = initial_logits + torch.tensor(np.random.rand(initial_logits.shape[0]))  # Update logits with random values
    updated_beta = initial_beta + update_beta
    
    current_distribution = gibbs_distribution_for_categories(categories, updated_logits, updated_beta)  # Calculate current distribution
    
    # Sample from the current distribution
    sample = current_distribution.sample()
    print(f"Sample at step: Category {categories[sample]}")

    return sample