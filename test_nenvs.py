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
    train_steps = 100  # Total number of training steps
    steps_per_design = 10  # Number of steps for each design
    batch_size = 10  # Batch size for design updates
    update_period = 5  # Number of designs before each update
    nenv = 2  # Number of parallel environments
    seed = 42  # Random seed for reproducibility
    with wandb.init(
        project='rl_whisker',
        id=run_id+'_train', group=run_id,
        job_type='train', resume='allow'
    ):
        env_id = "whisker-v0"
        num_cpu = 2  # Number of processes to use
        # Create the vectorized environment
        design_space = gym.spaces.Discrete(n=9)
        env = SubprocVecEnv([make_env(env_id,i, 1, 1000, config={"render": 1}) for i in range(num_cpu)])
        env = DesignManager(env,design_space=design_space,) # Wrap the environment in the DesignManager class
        env.configure_design_manager(steps_per_design=steps_per_design, maxlen=10)

        obs = env.reset()
        timer = 0
        for i in range(3):
            while True:
                timer +=1
                print(timer)
                actions = [env.action_space.sample() for _ in range(num_cpu)]
                obs, rewards, dones, info = env.step(actions)
                # print(obs)
                # print(rewards)
                # print(dones)
                # print(info)
                env.render()
                if timer >=steps_per_design:
                    
                    if i <= 1:
                        print("Resetting")
                        env.reset()
                    else: 
                        print("End")
                    timer = 0
                    break

            
            

