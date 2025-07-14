import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product
import gym
import wandb
import math as m
from coopt.distribution import RobotDesignDist,RobotDesignDistMixtureMVN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RunningMeanStd:
    def __init__(self):
        self.mean = 0.0
        self.S = 0.0      # running sum of squared devs
        self.count = 0

    def update(self, x):
        # x can be a list of new samples; here we do one at a time
        for r in x:
            self.count += 1
            delta = r - self.mean
            self.mean += delta / self.count
            delta2 = r - self.mean
            self.S += delta * delta2

    @property
    def variance(self):
        return self.S / self.count if self.count > 0 else 0.0

    @property
    def std(self):
        return m.sqrt(self.variance)
    



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

# --- RobotDesignOptimizer Class ---
class RobotDesignOptimizer(nn.Module):
    def __init__(self,
                 cut_off_length,
                 design_space,
                 ent_decay_start,
                 ent_decay_end):
        """
        Optimizer for joint robot design.
        
        Parameters:
          - discrete_values: List of discrete combinations (e.g. [(1, 'A'), (2, 'B'), ...]).
          - init_pressure_means: Tensor of shape (N, 3) initial means.
          - init_pressure_stds: Tensor of shape (N, 3) initial standard deviations.
          - ent_decay_start, ent_decay_end: Steps for entropy schedule.
        """
        super().__init__()
        self.cut_off_length = cut_off_length
        self.no_chamber = design_space['no_chamber']
        self.chamber_length = design_space['chamber_length']
        self.thickness = design_space['thickness']
        self.pressure_range= design_space['pressure_range']
        self.exploration_end_t = ent_decay_end
        #number of possible combination of body parameters
        self.discrete_combinations = list(product(self.no_chamber, self.chamber_length,self.thickness))
        self.N = len(self.discrete_combinations)
        self.reward_rms = [RunningMeanStd() for _ in range(self.N)]
        self.D = len(self.no_chamber)
        # Initialize discrete scores (which, when multiplied by beta, form logits).
        self.scores = torch.nn.Parameter(torch.zeros((self.N,), dtype=torch.float32), requires_grad=False)
        # Continuous parameters (not updated by gradients but via update() method).
        continuous_means_np = np.random.uniform(low=self.pressure_range[0], high=self.pressure_range[1], size=(self.N, self.D))
        self.continuous_means = torch.tensor(continuous_means_np, dtype=torch.float32)
        self.continuous_stds = torch.full((self.N , self.D), self.pressure_range.mean().item()/4) 
        self.target = LinearSchedule(np.log(self.N), 0, ent_decay_start, ent_decay_end)
        self.std_target = LinearSchedule(self.pressure_range.mean().item()/4, 0.01, ent_decay_start, ent_decay_end)
        self.beta = 0.0
        self.beta_min = 0.0
        self.beta_max = 5.0
        self.init_list_sample = 0
        self.pressure_grid_idx = 0
        self.baseline = torch.zeros(len(self.discrete_combinations), dtype=torch.float)
        self.beta_b   = 0.2            # smoothing factor for baseline
        self.beta_b_target = LinearSchedule(0.15, 0.05, ent_decay_start, ent_decay_end)
        self.alpha_target = LinearSchedule(0.2, 0.01, ent_decay_start, ent_decay_end)
        wandb.define_metric(f"design_{self.cut_off_length}/*", step_metric="train/step")

        # --- Grid phase setup ---
        p_min, p_max = float(self.pressure_range[0]), float(self.pressure_range[1])
        n_grid_steps=8
        pressure1_range = np.linspace(p_min, p_max, n_grid_steps)
        pressure2_range = np.linspace(p_min, p_max, n_grid_steps)
        self.pressure_grid = [(p1, p2) for p1 in pressure1_range for p2 in pressure2_range]  # size n_grid_steps^2

    def get_design_dist(self,type = 'single_variance'):
        # The discrete logits are computed as beta * scores.
        if type == 'single_variance':
            return RobotDesignDist(discrete_logits=self.beta * self.scores,
                                pressure_range = self.pressure_range,
                                continuous_means=self.continuous_means,
                                continuous_stds=self.continuous_stds,
                                discrete_values=self.discrete_combinations)
        elif type == 'covariance':
            return RobotDesignDistMixtureMVN(discrete_logits=self.beta * self.scores,
                                pressure_range = self.pressure_range,
                                continuous_means=self.continuous_means,
                                continuous_stds=self.continuous_stds,
                                discrete_values=self.discrete_combinations)

    def set_beta(self, t):
        high = self.beta_max
        low = self.beta_min 
        target = self.target(t)
        beta = (high + low) / 2.0
        dist = self.get_design_dist()  # with current beta (temporarily using beta from outer scope)
        # Compute entropy of the discrete part.
        ent = torch.distributions.Categorical(logits=dist.discrete_logits).entropy()
        # print(f"self.scores = {self.scores}")
        print(f"UPDATING ENTROPY FROM {ent} TO TARGET {target}")
        check = 0
        temp_beta = []
        # while torch.abs(ent - target) > 0.01 and check< 10:
            # check += 1
        while torch.abs(ent - target) > 0.0001:
            check += 1
            if check < 50:
                if ent > target:
                    low = beta
                else:
                    high = beta
                if abs(beta - (high + low)/2) < 0.00001:
                    if beta > (high + low)/2:
                        low = self.beta_min
                    elif beta < (high + low)/2:
                        high = self.beta_max
                beta = (high + low) / 2.0
                if beta > 0.99 * self.beta_max:
                    beta = self.beta_max
                    break
                # Update distribution temporarily.
                temp_logits = beta * self.scores
                
                ent = torch.distributions.Categorical(logits=temp_logits).entropy()
                temp_beta.append(beta)
            else:
                print("CHECK beta updating = ", beta)
                print(f"self.scores = {self.scores}")
                print("Temporary logits = ", temp_logits)
                print(f'Temporary beta = {temp_beta}')
                print("CHECK ent updating = ", ent)
                raise ValueError(f"Does not converge FROM {ent} TO TARGET {target}")
        self.beta = beta
        # print("UPDATED ENTROPY = ", ent)

    def sample(self,t):
        target = self.target(t)
        robot_dist = self.get_design_dist()
        joint      = robot_dist.get_distribution()  

        if abs(target - np.log(self.N)) <= 1e-4: 
            discrete_idx = self.init_list_sample
            self.init_list_sample = (discrete_idx + 1) % len(self.discrete_combinations)
        else:
            discrete_idx = int(joint.mixture_distribution.sample().item())
        # Discrete values
        discrete_values = self.discrete_combinations[discrete_idx]
        # Continuous values
        component_dist = joint.component_distribution
        transformed_pressure_list = component_dist.sample()[discrete_idx]
        inversed_pressure_list = transformed_pressure_list
        
        for transform in reversed(self.transformed_cont_dist.transforms):
            inversed_pressure_list = transform.inv(inversed_pressure_list)

        return {'no_chamber': discrete_values[0],
                'chamber_length': discrete_values[1],
                'thickness': discrete_values[2],
                'pressure1': torch.clamp(inversed_pressure_list[0], min=self.pressure_range[0],max=self.pressure_range[1]),
                'pressure2': torch.clamp(inversed_pressure_list[1], min=self.pressure_range[0],max=self.pressure_range[1])} ### pressure_list is 'list' type then there is no .numpy()
    
    def update_normal_dist(self, idx, sample, reward,t,max_std=0.15):
        """
        sample : tensor shape (2,)          # [p1, p2] sampled pressures
        reward : float
        """
        # Update running mean/std with the new reward
        self.reward_rms[idx].update([reward])

        # Compute normalized advantage
        raw_adv = reward - self.baseline[idx]
        denom = max(self.reward_rms[idx].std, 1e-3)   # or 1e-2
        adv   = raw_adv/ denom

        alpha = self.alpha_target(t)
        mu     = self.continuous_means[idx]     # (2,)
        sigma  = self.continuous_stds[idx]      # (2,)
        sigma = sigma.clamp(min=1e-3,max=max_std)          # avoid divide-by-zero
        sample = torch.as_tensor(sample, dtype=mu.dtype, device=mu.device)

        delta_mu    =  alpha * adv * (sample - mu)
        delta_sigma =  alpha * adv * ((sample - mu)**2 - sigma**2) / (sigma+ 1e-8)

        self.continuous_means.data[idx] += delta_mu
        self.continuous_stds.data[idx]  += delta_sigma

        scheduled_std = self.std_target(t)
        self.continuous_stds.data[idx].clamp_(min=scheduled_std, max=max_std)
        # Moving baseline per design class
        beta_b = self.beta_b_target(t)
        self.baseline[idx] = (1 - beta_b) * self.baseline[idx] + beta_b * reward

    def update(self, designs, rewards, t, k_top = 1):
        batch_disc_rewards = {}
        batch_cont_samples = {}

        for sample, reward in zip(designs, rewards):
            d_val = (sample['no_chamber'], sample['chamber_length'], sample['thickness'])
            try:
                idx = self.discrete_combinations.index(d_val)
            except ValueError:
                raise ValueError(f"Design value {d_val} not found in {self.discrete_combinations}")

            cont_value = torch.tensor([
                sample['pressure1'],
                sample['pressure2']
            ], dtype=torch.float)

            # Gather all samples and rewards for this idx
            if idx not in batch_disc_rewards:
                batch_disc_rewards[idx] = []
                batch_cont_samples[idx] = []
            batch_disc_rewards[idx].append(reward)
            batch_cont_samples[idx].append(cont_value)

        # Update continuous distributions using only k-top samples per idx
        for idx in batch_disc_rewards:
            rewards_tensor = torch.tensor(batch_disc_rewards[idx], dtype=torch.float)
            cont_tensor = torch.stack(batch_cont_samples[idx])  # shape: (N, 2)

            # Get indices of top-k rewards
            if len(rewards_tensor) > k_top:
                topk = torch.topk(rewards_tensor, k_top)
                top_indices = topk.indices
            else:
                top_indices = torch.arange(len(rewards_tensor))

            top_rewards = rewards_tensor[top_indices]
            top_cont = cont_tensor[top_indices]
            # Update discrete score (mean of top-k rewards)
            self.scores.data[idx] = float(top_rewards.mean())
            # Update continuous distribution for this idx using top-k only
            for c_sample, r in zip(top_cont, top_rewards):
                self.update_normal_dist(idx, c_sample, r, t)
            
        # Adjust the temperature parameter beta based on the updated scores.
        self.set_beta(t)


    def log(self, t):

        dist = self.get_design_dist()  # with current beta (temporarily using beta from outer scope)
        # Compute entropy of the discrete part.
        ent = torch.distributions.Categorical(logits=dist.discrete_logits).entropy()
        target = self.target(t)
        wandb.log({f'design_{self.cut_off_length}/beta': self.beta,
                   f'design_{self.cut_off_length}/ent': ent.item(),
                   f'design_{self.cut_off_length}/ent_target': target,
                #    f'design_{self.cut_off_length}/continuous_means': self.continuous_means,
                #    f'design_{self.cut_off_length}/continuous_stds': self.continuous_stds,
                #    f"design_{self.cut_off_length}/hist_gauss": wandb.Histogram(bins=xs, counts=counts),
                   f'design_{self.cut_off_length}/perplexity': np.exp(ent.item()),
                   f'design_{self.cut_off_length}/perplexity_target': np.exp(target),
                   f'train/step': t})
        wandb.log({
                "hist/all_means": wandb.Histogram(self.continuous_means.cpu().numpy().flatten()),
                "hist/all_stds": wandb.Histogram(self.continuous_stds.cpu().numpy().flatten()),
                "train/step": t
})
    
    def forward(self):
        return self.sample()

    def state_dict(self):
        return {'scores': self.scores,
                'continuous_means': self.continuous_means,
                'continuous_stds': self.continuous_stds,
                'beta': self.beta}

    def load_state_dict(self, state_dict):
        self.scores = state_dict['scores']
        self.continuous_means = state_dict['continuous_means']
        self.continuous_stds = state_dict['continuous_stds']
        self.beta = state_dict['beta']


# --- Example Testing ---
if __name__ == '__main__':
    wandb.init(project="robot_design_test")
    from sofagym.envs.Whisker.design_space.design_space import whiskerdesignspace
    no_chamber = torch.tensor([1,2,3])
    pressure_range = torch.tensor([0,0.001])
    body_length = torch.tensor([100, 80, 60])
    ins = whiskerdesignspace(body_length,no_chamber,pressure_range)
    design_optimizer = {}
    design_space = ins.design_space()
    for i in range(len(body_length)):
        space_idx = list(design_space.keys())[i]
        design_optimizer[i] = RobotDesignOptimizer(design_space[space_idx],
                                                        0, 
                                                        100)
    # key = list(design_space.keys())[0]
    # opt = RobotDesignOptimizer(design_space=design_space[key],
    #                            ent_decay_start=0,
    #                            ent_decay_end=100)
    
    # print(opt.sample())
    # for t in range(10):
    #     # Sample a batch of 100 designs.
    #     batch = [opt.sample() for _ in range(10)]
    #     print(batch)
    #     # Simulate rewards (for example, random rewards scaled by 10).
    #     rewards = 10 * torch.from_numpy(np.random.rand(100)).float()
    #     opt.update(batch, rewards, t)
    #     opt.log(t)
    #     if t % 10 == 0:
    #         print("check")
    #         state = opt.state_dict()
    #         # Emulate saving and reloading.
    #         new_opt = RobotDesignOptimizer(discrete_values,
    #                                        init_pressure_means,
    #                                        init_pressure_stds,
    #                                        ent_decay_start=0,
    #                                        ent_decay_end=100)
    #         new_opt.load_state_dict(state)
    #         opt = new_opt

    #     print("end")
