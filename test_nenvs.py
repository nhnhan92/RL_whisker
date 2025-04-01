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
    total_steps = 1000
    n_steps = 20  # Total number of training steps for each sampling time
    batch_size = 5  # Batch size for design updates
    update_period = 3  # Number of designs before each update
    nenv = 5  # Number of parallel environments
    seed = 42  # Random seed for reproducibility
    no_int = 5
    ent_decay_start = 20
    ent_decay_end = 1000
    no_chamber = torch.tensor([1,2])
    pressure_range = torch.tensor([0.00001,0.2])
    body_length = np.linspace(60,100,5,dtype=int).tolist()
    thickness = torch.tensor(np.linspace(2,4,5,dtype=float))
    chamber_length = torch.tensor(np.linspace(20,40,11,dtype=int))
    ins = whiskerdesignspace(body_length,no_chamber,chamber_length,thickness,pressure_range)
    with wandb.init(
        project='rl_whisker',
        id=run_id+'_train', group=run_id,
        job_type='train', resume='allow',reinit = True
    ):
        env_id = "whisker-v0"
        # Create the vectorized environment
        design_space = ins.design_space()
        env = SubprocVecEnv([make_env(env_id = env_id,
                                      rank = i, 
                                    #   seed = 1,
                                      max_episode_steps = n_steps,
                                        config={"render": 1}) for i in range(nenv)])
        env = DesignManager(env,design_space=design_space,
                            n_steps = n_steps,
                            n_env = nenv,
                            batch_size =batch_size,
                            update_period = update_period,
                            ent_decay_start = ent_decay_start,
                            ent_decay_end = ent_decay_end,
                            cut_off_list = body_length) # Wrap the environment in the DesignManager class

        obs = env.reset()
        timer = 0
        ite = 0
        while ite*n_steps<total_steps:
            timer +=1
            actions = [env.action_space.sample() for _ in range(nenv)]
            obs, rewards, dones, info = env.step(actions)
            # print(obs)
            # print(rewards)
            # print(dones)
            # print(info)
            # env.render()
            if timer == int(n_steps/nenv):
                
                print("Resetting")
                env.reset()
                timer = 0
                ite += 1

            
            

