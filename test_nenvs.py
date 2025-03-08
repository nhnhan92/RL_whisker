import gym
import numpy as np
from coopt.coopt_vec_env import SubprocVecEnv
# from stable_baselines3.common.vec_env import (SubprocVecEnv,
#                                               VecMonitor, VecVideoRecorder,
#                                               sync_envs_normalization)
from stable_baselines3.common.utils import set_random_seed
import sofagym
from sofagym.envs import *
from agents.utils import make_env
from coopt.design_manager import DesignManager
import wandb
import os
from sofagym.envs.Whisker.design_space.design_space import whiskerdesignspace
import torch
RANDOM = False

import psutil
pid = os.getpid()
py = psutil.Process(pid)

sys.path.insert(0, os.getcwd()+"/..")

__import__('sofagym')

import time

if __name__ == '__main__':
    logdir = './test_coopt'
    run_id = os.path.basename(logdir)
    total_steps = 100
    n_steps = 10  # Total number of training steps for each sampling time
    batch_size = 4  # Batch size for design updates
    update_period = 4  # Number of designs before each update
    nenv = 2  # Number of parallel environments
    seed = 42  # Random seed for reproducibility
    no_int = 4
    ent_decay_start = 1
    ent_decay_end = 64
    no_chamber = torch.tensor([1,2,3])
    pressure_range = torch.tensor([0,0.001])
    body_length = torch.tensor([100, 80, 60])
    ins = whiskerdesignspace(body_length,no_chamber,pressure_range)
    with wandb.init(
        project='rl_whisker',
        id=run_id+'_train', group=run_id,
        job_type='train', resume='allow'
    ):
        env_id = "whisker-v0"
        # Create the vectorized environment
        design_space = ins.design_space()
        env = SubprocVecEnv([make_env(env_id = env_id,
                                      rank = i, 
                                      seed = 1,
                                      max_episode_steps = n_steps,
                                        config={"render": 1}) for i in range(nenv)])
        env = DesignManager(env,design_space=design_space,
                            n_steps = n_steps,
                            n_env = nenv,
                            batch_size =batch_size,
                            update_period = update_period,
                            ent_decay_start = ent_decay_start,
                            ent_decay_end = ent_decay_end) # Wrap the environment in the DesignManager class

        obs = env.reset()
        timer = 0
        ite = 0
        while ite*n_steps<total_steps:
            timer +=1
            actions = [env.action_space.sample() for _ in range(nenv)]
            obs, rewards, dones, info = env.step(actions)
            # print(obs)
            print(rewards)
            print(dones)
            # print(info)
            env.render()
            if timer == int(n_steps/nenv) and ite < no_int:
                
                print("Resetting")
                env.reset()
                timer = 0
                ite += 1
            elif timer == int(n_steps/nenv) and ite >= no_int:
                print("End")
            
                break

            
            

