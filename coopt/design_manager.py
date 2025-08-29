import gym.spaces
import numpy as np
from collections import deque
from dl import nest
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
import torch
import wandb
import gym
import os
from dl.ckptr import Checkpointer
from coopt.robot_optimizer import RobotDesignOptimizer
from itertools import product, islice
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
class DesignLogger:
    """Records and keeps the history of designs and refwards."""

    def __init__(self, maxlen,cut_off_list,discrete_combinations,resume):
        self.maxlen = maxlen
        self.discrete_combinations = discrete_combinations
        self.resume = resume
        self.designs = {}
        self.rewards = {}
        self.count = {}
        self.data = {}
        self.data_post_processed = {}
        self.design_idx_top_score = {}
        self.count_disc_categories = {}
        self.body_length = cut_off_list
        for i in self.body_length :
            self.designs[i] = deque([], maxlen=maxlen)
            self.rewards[i] = deque([], maxlen=maxlen)
            self.count[i] = 0
            self.data[i] = []
            self.data_post_processed[i] = []
            self.design_idx_top_score[i] = {}
            self.count_disc_categories[i] = {}
            for j in range(len(self.discrete_combinations)):
                self.design_idx_top_score[i][j] = 0
                self.count_disc_categories[i][j] = 0
            wandb.define_metric(f"designs_{i}", step_metric="train/design_count")

        wandb.define_metric("train/design_count", step_metric="train/step")
        self.columns = ['count', 'design', 'reward']
        

    def log(self, design, reward,no_design: int = 100):
        body_length = design['body_length']
        reward = 0 if reward < 0 else reward
        self.designs[body_length].append(design)
        self.rewards[body_length].append(reward)
        self.count[body_length] += 1
        self.data_post_processed[body_length].append({'count':self.count[body_length],
                                                       'no_chamber': design['no_chamber'],
                                                       'chamber_length':design['chamber_length'],
                                                       'thickness':design['thickness'],
                                                       'pressure1':design['pressure1'],
                                                       'pressure2':design['pressure2'],
                                                       'reward':reward})
        
        d_val = (design['no_chamber'], design['chamber_length'], design['thickness'])
        idx = self.discrete_combinations.index(d_val)
        self.count_disc_categories[body_length][idx] += 1
        if self.design_idx_top_score[body_length][idx] < reward:
            self.design_idx_top_score[body_length][idx] = reward
        
        if self.count[body_length] % no_design == 0:
            design_list, reward_list = self.get_designs_and_rewards(no_design,body_length)  
            # 1. Zip them together into (reward, design) pairs
            paired = list(enumerate(zip(reward_list, design_list)))
            # 2. Sort the pairs by reward in descending order
            paired_sorted = sorted(paired, key=lambda x: x[1][0], reverse=True)
            top3 = list(islice(paired_sorted, 5))  
            columns = ["Design_count", "Reward", "Design"]
            for i, (orig_idx,(r, design)) in enumerate(top3):
                # Check the type of design.
                if isinstance(design, dict):    
                    # Convert dictionary into a string: "key1=value1, key2=value2, ..."
                    design_str = ", ".join(str(value) for value in design.values())
                elif isinstance(design, (list, tuple)):
                    design_str = str([int(x) for x in design])
                else:
                    design_str = str(int(design))
                self.data[body_length].append([(self.count[body_length] - no_design) + orig_idx+1, r, design_str])     
            tbl= wandb.Table(data=self.data[body_length], columns=columns)
            wandb.log({f"reward_tracking_{body_length}": tbl, "train/design_count": self.count[body_length]})
            # plot = wandb.plot_table(
            #         vega_spec_name = "wandb/line/v0",  # use wandb's default line plot template
            #         data_table = tbl,
            #         fields = {"x": "Design_count", "y": "Reward", "name": "Design"},  # this sets "Design" as legend/label
            #         string_fields={"title": "Reward tracking"},
            #         )
            # wandb.log({f"designs_{body_length}": plot, "train/design_count": self.count[body_length]})
            self.log_count_bar()
            # self.data[body_length].append([self.count[body_length], design_str, float(reward)])
        # if self.count[body_length] % 20 == 0:
        #     # dump_self_data_size(self.data, tag=f" #{self.count[body_length]}")
        #     table = wandb.Table(data=self.data[body_length], columns=self.columns)
        #     wandb.log({f"designs_{body_length}": table, "train/design_count": self.count[body_length]})
    def log_class_rank(self,extra_rewards, t):
        extra_rewards_list = [extra_rewards[i] for i in sorted(extra_rewards)]
        x_vals = np.arange(len(extra_rewards_list)).tolist()
        y_vals = [extra_rewards_list] 
        plot = wandb.plot.line_series(
                    xs=x_vals,
                    ys=y_vals,
                    keys=[f"Extra reward rate"],
                    title=f"Disc_var_rank",
                    xname="design idx")

        wandb.log({f"Disc_var_rank/Disc_var_rank_plot": plot,
                "train/step": t})
        
    def log_count_bar(self, section="Design_count_bar"):
        xs   = list(range(len(self.discrete_combinations)))
        for body_length in self.body_length:
            ys   = self.count_disc_categories[body_length].values()

            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(xs, ys, linestyle='None', marker='o', color='steelblue')
            # cosmetic: remove background, spines, etc.
            ax.set_facecolor("none")          # axes bg transparent
            fig.patch.set_alpha(0)            # figure bg transparent
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

            ax.set_xlabel("design idx")
            ax.set_ylabel("count")
            ax.set_title(f"Count per idx  (body_length={body_length})")
            # 2 ── log to W&B as an image ─────────────────────────────────────────
            wandb.log({f"{section}/count_bar_{body_length}": wandb.Image(fig),
                    "train/design_count": self.count[body_length]})
            plt.close('all')   # free the memory

    def state_dict(self,top_k: int = 5):
        # convert each deque to a list so it can be pickled
        designs_serialised = {k: list(v) for k, v in self.designs.items()}
        rewards_serialised = {k: list(v) for k, v in self.rewards.items()}
        sums = {}
        for body_length, inner in self.design_idx_top_score.items():
            for key, value in inner.items():
                # Here, key is the "second item" (e.g., '1', '2', '3').
                sums[key] = sums.get(key, 0) + value
        # -- pick “good” categories (top-K here) -------------------------------
        good_keys = sorted(sums, key=sums.get, reverse=True)[:top_k]
        good_vals = [sums[k] for k in good_keys]
        hi, lo = max(good_vals), min(good_vals)
        denom = hi - lo or 1.0           # avoid ÷0  (all equal → set to 0)
        extra_rewards = {k: (sums[k] - lo) / denom if k in good_keys else 0.0 for k in sums}
        # max_sum = max(sums.values())
        # min_sum = min(sums.values())
        # range = max_sum - min_sum
        # extra_rewards = {k: (v - min_sum) / range for k, v in sums.items()}
        
        return {
            'designs': self.designs,
            'rewards': self.rewards,
            'count': self.count,
            'extra_rewards': extra_rewards
        }

    def load_state_dict(self, state_dict):
        self.designs = state_dict['designs']
        self.rewards = state_dict['rewards']
        self.count = state_dict['count']

    def get_designs_and_rewards(self, n,cut_off_length):
        return list(self.designs[cut_off_length])[-n:], list(self.rewards[cut_off_length])[-n:]

    def get_design_count(self):
        return self.count
# --- LinearSchedule Class (as before) ---
class LinearSchedule():
    def __init__(self, start_val, end_val, start_step, end_step):
        self.sv = start_val
        self.ev = end_val
        self.ss = start_step
        self.es = end_step

    def __call__(self, t):
        if t <= self.ss:
            return self.sv
        if t >= self.es:
            return self.ev
        frac = (t - self.ss) / (self.es - self.ss)
        return self.ev * frac + self.sv * (1 - frac)

class DesignManager(VecEnvWrapper):
    """Environment wrapper to handle design sampling/logging.

    This wrapper:
         - samples and sets design parameters at fixed intervals.
         - records and logs the performance of each design.
    """
    
    def __init__(self, venv, design_space=None,
                 n_steps = 100,
                 n_env = 1,
                 batch_size = 0,
                 update_period = 10,
                 ent_decay_start = 1,
                 ent_decay_end = 10,
                 cut_off_list = [60,70,80,90],
                 test_mode=False,
                 shared_distribution = None,
                 shared_logger = None,
                 resume = False,
                 logdir = None,
                 save_freq = 300):
        super().__init__(venv)
        self.shared_logger = shared_logger
        self.test_mode = test_mode
        self.resume = resume
        self.n_env = n_env
        self.steps_per_design = n_steps # Number of evironment steps each design takes
        self.design_count = 0
        self.batch_size = batch_size # number of designs used in each update
        self.steps = 0
        self.t = 0
        self.update_period = update_period  #number of sampled designs that reach before dist update
        self.cut_off_list = cut_off_list
        self.design_space = design_space
        self.designs = None
        self.rewards = np.zeros(self.n_env)
        self.logdir = logdir
        self.ent_decay_start = ent_decay_start # start of entropy decay
        self.ent_decay_end = ent_decay_end  # end of entropy decay
        self.save_freq = save_freq
        self.design_optimizer = {}
        self.last_design_update = {}
        self.design_avg_extra_reward = LinearSchedule(0, 2, ent_decay_end - 500000, ent_decay_end) # -> only give 500k step before training ends
        self.discrete_combinations = list(product(self.design_space[self.cut_off_list[0]]['no_chamber'], 
                                                  self.design_space[self.cut_off_list[0]]['chamber_length'],
                                                  self.design_space[self.cut_off_list[0]]['thickness']))
        if not self.test_mode:
            for i in self.cut_off_list:
                self.last_design_update[i] = 0
                self.design_optimizer[i] = RobotDesignOptimizer(i,
                                                                self.design_space[i],
                                                                self.ent_decay_start, 
                                                                self.ent_decay_end)
        else:
            self.design_optimizer = shared_distribution
        if self.logdir:
            self.ckptr = Checkpointer(os.path.join(self.logdir, 'ckpts_design_params'))
        self.configure_design_manager(self.batch_size)
        if self.resume:
            # try to load immediately (when env is created outside SB3)
            self.load()
    def distribution_state_reference(self):
        shared_dist = []
        for cut_off_length, optimizer in self.design_optimizer.items():
            shared_dist[cut_off_length] = optimizer
        return shared_dist

    def set_design_dist(self, design_dist):
        self.design_dist = design_dist
        
    def get_design_count(self):
        return self.logger.get_design_count()

    def configure_design_manager(self, maxlen):
        if not self.test_mode:
            self.logger = DesignLogger(maxlen = maxlen,
                                        cut_off_list=self.cut_off_list,
                                        discrete_combinations = self.discrete_combinations,
                                        resume=self.resume)
        else:
            self.logger = self.shared_logger

    def get_designs_and_rewards(self, n,cut_off_length):
        return self.logger.get_designs_and_rewards(n,cut_off_length)

    def _unnorm(self, p):
        # space = self.observation_space['design']
        # if isinstance(space, gym.spaces.Box):
        #     return 0.5 * (p.clamp_(-1., 1.) + 1.) * (space.high - space.low) + space.low
        # else:
        #     return p
        return p
    
    def _sample_design(self,optimizer):
        with torch.no_grad():
            return self._unnorm(nest.map_structure(
                            lambda x: x.numpy().item(), optimizer.sample(self.t)))

    def _sample_mode(self,optimizer):
        with torch.no_grad():
            return self._unnorm(nest.map_structure(
                            lambda x: x.numpy(), optimizer.mode()))

    def init_scene_with_mode(self):
        self.designs = [self._sample_mode() for _ in range(self.n_env)]
        self.venv.set_designs(self.designs)
        self.rewards = np.zeros(self.n_env)

    def init_scene(self):
        self.designs = []
        n_design_per_length = self.n_env // len(self.cut_off_list)
        if not self.test_mode:
            for cut_off_length, optimizer in self.design_optimizer.items():
                for _ in range(n_design_per_length):
                    sample = self._sample_design(optimizer = optimizer)  # sample is expected to be a dict.
                    sample = {'body_length': cut_off_length, **sample}
                    self.designs.append(sample)
            for cut_off_length, optimizer in self.design_optimizer.items():
                if len(self.designs) < self.n_env:
                    sample = self._sample_design(optimizer = optimizer)  # sample is expected to be a dict.
                    sample = {'body_length': cut_off_length, **sample}
                    self.designs.append(sample)
        # self.designs = [self._sample_design() for _ in range(self.num_envs)]
        # print("designs: ", self.designs
        else:
            for i in self.cut_off_list:
                designs, rewards = self.get_designs_and_rewards(self.batch_size,i)
                paired = list(enumerate(zip(rewards, designs)))
                # 2. Sort the pairs by reward in descending order
                paired_sorted = sorted(paired, key=lambda x: x[1][0], reverse=True)
                top_design_tested= list(islice(paired_sorted, self.n_env))  
                for _, (orig_idx,(r, design)) in enumerate(top_design_tested):
                    self.designs.append(design)
        self.venv.set_designs(self.designs)
        self.rewards = np.zeros(self.n_env)
    
    def reset(self):
        if self.designs is None:
            print("self.designs is None")
            self.init_scene()
        return self.venv.reset()

    def trigger_dist_update(self):
        # Update the design distribution
        design_count = self.get_design_count()
        print("Design Count =", design_count)
        state_dict = self.state_dict()
        extra_rewards = state_dict['extra_rewards']
        # print(extra_rewards)
        base_extra_reward = self.design_avg_extra_reward(self.t)
        # print(f'base_extra_reward = {base_extra_reward}')
        for i in self.cut_off_list:
            designs_since_update = design_count[i] - self.last_design_update[i]
            if designs_since_update >= self.update_period:
                designs, rewards = self.get_designs_and_rewards(self.batch_size,i)
                if len(self.cut_off_list) > 1:
                    for j, (sample,reward) in enumerate(zip(designs, rewards)):
                        d_val = (sample['no_chamber'], sample['chamber_length'], sample['thickness'])
                        idx = self.discrete_combinations.index(d_val)
                        rewards[j] += base_extra_reward * extra_rewards[idx]
                self.design_optimizer[i].update(designs, rewards,
                                            self.t)
                self.design_optimizer[i].log(self.t)
                self.last_design_update[i] = design_count[i]
        # if designs_since_update >= self.update_period:
        #     self.logger.log_class_rank(extra_rewards,self.t)
    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        # Log the current step
        if self.t == 0:
            for i in self.cut_off_list:
                self.design_optimizer[i].log(self.t)

        obs, rewards, dones, infos = self.venv.step_wait()
        dt = self.n_env
        self.t += dt
        self.steps += 1
        self.rewards += rewards
        if self.steps_per_design and self.steps % self.steps_per_design == 0:
            if not self.test_mode:
                for design, reward in zip(self.designs, self.rewards):
                    self.logger.log(design, reward)
                self.trigger_dist_update()
            self.init_scene()
            dones[:] = True  # Force reset after a design interval
            if not self.test_mode and self.t%self.save_freq == 0 and self.t != 0 :
                self.save()
        return obs, rewards, dones, infos
    def state_dict(self):
        return self.logger.state_dict()

    def load_state_dict(self, state_dict):
        return self.logger.load_state_dict(state_dict['designs'])

    def save(self):
        # state = self.design_optimizer.state_dict()
        # self.ckptr.save(state, self.t)
        dist_state = {cut: opt.state_dict()
                  for cut, opt in self.design_optimizer.items()}
        payload = {
            "dist_state": dist_state,
            "logger_state": self.logger.state_dict(),
            "design_data":self.logger.data_post_processed,
            "step": self.t
        }
        if self.logdir:
            self.ckptr.save(payload, self.t)

    # def load(self, t=None):
    #     state_dict = self.ckptr.load(t)
    #     if state_dict is not None:
    #         self.design_optimizer.load_state_dict(state_dict)
    #     design_count = self.get_design_count()
    #     self.last_design_update = design_count - (design_count % self.update_period)
    def load(self, t=None):
        payload = self.ckptr.load(t)
        if payload is None:
            return

        dist_state = payload["dist_state"]
        for cut, sd in dist_state.items():
            self.design_optimizer[cut].load_state_dict(sd)

        # restore logger
        self.logger.load_state_dict(payload["logger_state"])

        # restore counters
        self.t = payload.get("step", 0)
        design_count = self.get_design_count()
        self.last_design_update = {
            cut: design_count[cut] - design_count[cut] % self.update_period
            for cut in self.cut_off_list
        }
