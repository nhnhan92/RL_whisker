from coopt.coopt_vec_env import SubprocVecEnv
import wandb
import gym
import sofagym
from sofagym.envs import *
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
import argparse
from agents.RLberryAgent import RLberryAgent
from agents.SB3Agent import SB3Agent
from agents.utils import args_check

results_dir = "./Results"

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


def test_trainer_sofagym():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--algorithm", help = "RL algorithm",
                        type=str, required=False, default="PPO")
    parser.add_argument("-fr", "--framework", help = "RL framework",
                        type=str, required=False, default='SB3')
    parser.add_argument("-ne", "--env_num", help = "Number of parallel envs",
                        type=int, required=False, default=1)
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

    env_name = "whisker-v0"
    algo_name = args.algorithm
    args_check(algo_name, algos, 'algorithm')
    framework = args.framework
    args_check(framework, frameworks, 'framework')

    n_envs = args.env_num
    seed = args.seed
    total_timesteps = args.total_timesteps
    max_episode_steps = args.max_steps
    train = args.train
    test = args.test
    n_tests = args.num_test
    model_dir = args.model_dir
    
    if model_dir is None:
        if train == 'continue' or (train == 'none' and test):
            parser.error("Valid argument --model_dir must be provided where previous model training files are saved")
    
    # Agent = eval(framework + "Agent")
    Agent = eval(framework + "Agent")
    if train == 'new':
        logdir = './test_coopt'
        run_id = os.path.basename(logdir)
        with wandb.init(project='rl_whisker',
                        id=run_id+'_train', group=run_id,
                        job_type='train', resume='allow'):
                
            nenv = 2  # Number of processes to use
            seed = 1 # The inital seed for RNG
            design_space = gym.spaces.Discrete(n=9)
            env = SubprocVecEnv([make_env(env_name, rank=i, seed=seed, max_episode_steps=max_episode_steps, config={"render": 1}) for i in range(nenv)]) # Create the vectorized environment
            env = DesignManager(env, design_space) # Wrap the environment in the DesignManager class
                
            steps_per_design = 100 # Number of evironment steps each design takes
            batch_size = 32 # number of designs used in each update
            update_period = 2 # number of designs sampled between updates
            train_steps = 100 # number of training steps
            ent_decay_start = 0 # start of entropy decay
            ent_decay_end = 10000 # end of entropy decay
            co_opt = CoOpt(env, design_space, logdir, steps_per_design, batch_size, update_period, ent_decay_start, ent_decay_end)
            
            agent = Agent(env,env_name, algo_name, seed, results_dir, max_episode_steps, n_envs)

            # while co_opt.t < train_steps:
            agent.fit(total_timesteps = steps_per_design)
                # co_opt.step()
                # env.render()

            # co_opt.save()
            
    else:
        agent = Agent.load(model_dir)
        
        if train == 'continue':
            agent.fit(total_timesteps)

    if test:
        agent.eval(n_tests, model_timestep='best_model', render=True, record=True)

    agent.close()
    print("... End.")


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# Create an RL agent
def create_agent(env, logdir):
    return PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=logdir
    )


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
        design_space = gym.spaces.Discrete(n=9)
        env = SubprocVecEnv([make_env(env_id, rank=i, seed=seed, max_episode_steps=1000, config={"render": 1}) for i in range(nenv)]) # Create the vectorized environment
        env = DesignManager(env, design_space) # Wrap the environment in the DesignManager class

        for i in range(4):
            print(f"Episode {i} Starting")
            
            steps_per_design = 10 # Number of evironment steps each design takes
            batch_size = 32 # number of designs used in each update
            update_period = 2 # number of designs sampled between updates
            train_steps = 100 # number of training steps
            ent_decay_start = 0 # start of entropy decay
            ent_decay_end = 10000 # end of entropy decay
            co_opt = CoOpt(env, design_space, logdir, steps_per_design, batch_size, update_period, ent_decay_start, ent_decay_end)
            while co_opt.t < train_steps:
            # while True:
                co_opt.step()
                env.render()

            env.reset()
        # co_opt.save()

def train_co_opt():
    """
    Function to train the co-optimization framework with reinforcement learning.
    """
    logdir = './test_coopt'
    run_id = os.path.basename(logdir)
    train_steps = 100  # Total number of training steps
    steps_per_design = 5  # Number of steps for each design
    batch_size = 10  # Batch size for design updates
    update_period = 5  # Number of designs before each update
    nenv = 2  # Number of parallel environments
    seed = 42  # Random seed for reproducibility

    with wandb.init(
        project='rl_whisker',
        id=run_id+'_train', group=run_id,
        job_type='train', resume='allow'
    ):
        # Set up environment
        env_id = "whisker-v0"
        design_space = gym.spaces.Discrete(n=9)
        vec_env = SubprocVecEnv([
            make_env(env_id, rank=i, seed=seed, max_episode_steps=50, config={"render": 1})
            for i in range(nenv)
        ])
        env = DesignManager(vec_env, design_space)
        # env.configure_design_manager(steps_per_design, maxlen=10)
        
        # Initialize CoOpt framework
        co_opt = CoOpt(
            env=env,
            design_space=design_space,
            logdir=logdir,
            steps_per_design=steps_per_design,
            batch_size=batch_size,
            update_period=update_period,
            ent_decay_start=0,
            ent_decay_end=train_steps
        )
        
        # Initialize RL agent
        agent = create_agent(env, logdir)
        
        # Training loop
        for i in range(10):
            print(f"i = {i}")
            # Train RL agent
            agent.learn(total_timesteps=200)
            # Step CoOpt framework
            # co_opt.step()
        
        # Save the final RL model and co-opt state
        agent.save(os.path.join(logdir, "final_model"))
        co_opt.save()

if __name__ == '__main__':
    test_co_opt()
    # test_trainer_sofagym()
    # train_co_opt()