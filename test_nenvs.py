import gym
import numpy as np

from stable_baselines3.common.vec_env import (SubprocVecEnv,
                                              VecMonitor, VecVideoRecorder,
                                              sync_envs_normalization)
from stable_baselines3.common.utils import set_random_seed
import sofagym
from sofagym.envs import *
from agents.utils import make_env
RANDOM = False

import psutil
pid = os.getpid()
py = psutil.Process(pid)

sys.path.insert(0, os.getcwd()+"/..")

__import__('sofagym')

import time

if __name__ == '__main__':
    env_id = "whisker-v0"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    
    env = SubprocVecEnv([make_env(env_id, 0, 1, 1000, config={"render": 1}) for _ in range(num_cpu)])
    # actions = [env.action_space.sample() for _ in range(num_cpu)]
    obs = env.reset()
    # print("======================================")
    # env.step(actions)
    
    # print(obs)

    while True:
        actions = [env.action_space.sample() for _ in range(num_cpu)]
        obs, rewards, dones, info = env.step(actions)
        print(obs)
        print(rewards)
        print(dones)
        print(info)
        env.render()
        if any(dones):
            break