# -*- coding: utf-8 -*-
"""Train SOFA scenes as gym environments using RL algorithms.

Usage:
-----
    python3 rl.py -e env_id -a algo_name
"""

__authors__ = "emenager, pschegg"
__contact__ = "etienne.menager@ens-rennes.fr, pierre.schegg@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020,Inria"
__date__ = "Nov 10 2020"


import argparse
from agents.RLberryAgent import RLberryAgent
from agents.SB3Agent import SB3Agent
from agents.utils import args_check
import sofagym
from sofagym.envs import *
import wandb
import os
from sofagym.envs.Whisker.design_space.design_space import whiskerdesignspace
import torch
import numpy as np
import yaml
from pathlib import Path
import datetime
torch.cuda.empty_cache() 
envs = {
        1: 'bubblemotion-v0',
        2: 'cartstem-v0',
        3: 'cartstemcontact-v0',
        4: 'catchtheobject-v0',
        5: 'concentrictuberobot-v0',
        6: 'diamondrobot-v0',
        7: 'gripper-v0',
        8: 'maze-v0',
        9: 'multigaitrobot-v0',
        10: 'simple_maze-v0',
        11: 'stempendulum-v0',
        12: 'trunk-v0',
        13: 'trunkcup-v0',
        14: 'cartpole-v0',
        15: 'catheter_beam-v0',
        16: 'whisker-v0'
        }

algos = {
        1: 'SAC',
        2: 'PPO',
        3: 'A2C',
        4: 'DQN',
        5: 'TD3',
        6: 'DDPG',
        7: 'REINFORCE'
        }

frameworks = {
        1: 'SB3',
        2: 'RLberry'
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", help="Name of the environment",
                        type=str, required=True)
    parser.add_argument("-a", "--algorithm", help = "RL algorithm",
                        type=str, required=False, default='PPO')
    parser.add_argument("-ne", "--env_num", help = "Number of parallel envs",
                        type=int, required=False, default=1)
    parser.add_argument("-fr", "--framework", help = "RL framework",
                        type=str, required=False, default='SB3')
    parser.add_argument("-s", "--seed", help = "Seed",
                        type=int, required=False, default=0)
    parser.add_argument("-st", "--total_timesteps", help = "Number of training timesteps",
                        type=int, required=False, default=None)
    parser.add_argument("-mst", "--max_steps", help = "Max steps per episode",
                        type=int, required=False, default=None)
    parser.add_argument("-tr", "--train", help = "Training a new model or continue training from saved model",
                        choices=['new', 'continue', 'none'], required=False, default='new')
    parser.add_argument("-te", "--test", help = "Testing flag",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-tn", "--num_test", help = "Number of tests",
                        type=int, required=False, default=1)
    parser.add_argument("-md", "--model_dir", help = "Model directory",
                        type=str, required=False, default=None)
    
    args = parser.parse_args()

    env_name = args.environment
    args_check(env_name, envs, 'environment')
    algo_name = args.algorithm
    args_check(algo_name, algos, 'algorithm')
    framework = args.framework
    args_check(framework, frameworks, 'framework')

    n_envs = args.env_num
    seed = args.seed # Random seed for reproducibility
    train = args.train
    test = args.test
    n_tests = args.num_test
    model_dir = args.model_dir
    parent_dict = os.path.dirname(os.path.abspath(__file__))
    model_dir = parent_dict + "/Results/whisker-v0/PPO/whisker_rl/"
    model_name = 'whisker_rl'
    logdir = './test_coopt'
    results_dir = "./Results"
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    total_steps = args.total_timesteps
    max_episode_steps = args.max_steps  # Total number of training steps for each sampling time (episode)
    kwargs = {'model_params':
                {'params_path': parent_dict+"/sofagym/envs/Whisker/whisker_params.yml"}}

    with wandb.init(project='rl_whisker',
                    id=f"train_{run_id}", group=f"session_{run_id}",
                    job_type='train', resume='allow'):
        if model_dir is None:
            if train == 'continue' or (train == 'none' and test):
                parser.error("Valid argument --model_dir must be provided where previous model training files are saved")
        
        Agent = eval(framework + "Agent")
        if train == 'new':
            agent = Agent(env_id = env_name,
                            algo_name = algo_name,
                            seed=seed, 
                            output_dir = results_dir, 
                            max_episode_steps=max_episode_steps, 
                            n_envs = n_envs,
                            model_name = model_name,
                            **kwargs
                            )
            agent.fit(total_steps)
        else:
            agent = Agent.load(model_dir)
            
            if train == 'continue':
                agent.fit(total_steps)

        if test:
            agent.eval(n_tests, model_timestep='best_model', render=True, record=True)

    agent.close()
    print("... End.")
