from coopt.coopt_vec_env import SubprocVecEnv
from sofagym.envs import *
import wandb
import gym
from agents.utils import make_env
from coopt.design_manager import DesignManager
import os
RANDOM = False

import psutil
pid = os.getpid()
py = psutil.Process(pid)

sys.path.insert(0, os.getcwd()+"/..")

__import__('sofagym')


from coopt.trainer import CoOpt



def test_co_opt():
    """
    Function to test the co-optimization algorithm.

    This function creates a vectorized environment, initializes the co-optimization algorithm,
    and runs the optimization loop until a certain number of steps is reached.

    Args:
        None

    Returns:
        None
    """

    logdir = './test_coopt'
    run_id = os.path.basename(logdir)
    with wandb.init(project='rl_whisker',
                    id=run_id+'_train', group=run_id,
                    job_type='train', resume='allow'):
            
        env_id = "whisker-v0"
        nenv = 2  # Number of processes to use
        seed = 1 # The inital seed for RNG
        design_space = gym.spaces.Discrete(n=3)
        env = SubprocVecEnv([make_env(env_id, rank=i, seed=seed, max_episode_steps=1000, config={"render": 1}) for i in range(nenv)]) # Create the vectorized environment
        env = DesignManager(env, design_space) # Wrap the environment in the DesignManager class
        env.reset()

        steps_per_design = 10 # Number of evironment steps each design takes
        batch_size = 32 # number of designs used in each update
        update_period = 1 # number of designs sampled between updates
        train_steps = 1000 # number of training steps
        ent_decay_start = 0 # start of entropy decay
        ent_decay_end = 10000 # end of entropy decay
        co_opt = CoOpt(env, design_space, logdir, steps_per_design, batch_size, update_period, ent_decay_start, ent_decay_end)
        while co_opt.t < train_steps:
            co_opt.step()
        co_opt.save()

if __name__ == '__main__':
    test_co_opt()