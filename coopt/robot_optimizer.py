import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product
import gym
import wandb
from dl.distributions import CatDist  # If needed; here we rely on our new RobotDesignDist
# --- RobotDesignDist Class ---
# This class simply outputs the joint distribution.
# (It is adapted to accept as input tensors computed externally, e.g. logits = reward*(1/temperature))
class RobotDesignDist(nn.Module):
    """
    Joint distribution for robot design parameters.
    
    A robot design is parameterized by 5 elements:
      - Two discrete parameters (represented as a tuple, e.g. (d1, d2))
      - Three continuous parameters.
    
    The joint probability is modeled as:
    
         P(robot design) = P(discrete) * P(continuous | discrete)
    
    Inputs:
      - discrete_logits: Tensor of shape (N,), where N is the number of discrete combinations.
      - continuous_means: Tensor of shape (N, 3) giving the mean for the 3 continuous parameters per combination.
      - continuous_stds: Tensor of shape (N, 3) giving the std for the 3 continuous parameters per combination.
      - discrete_values: List of length N of tuples (each tuple is one discrete combination).
    """
    def __init__(self, discrete_logits,pressure_range, continuous_means, continuous_stds, discrete_values):
        super().__init__()
        self.discrete_logits = discrete_logits
        self.continuous_means = continuous_means
        self.continuous_stds = continuous_stds
        self.discrete_values = discrete_values
        self.pressure_range = pressure_range

    def get_distribution(self):
        # Build the discrete distribution.
        self.mixture_dist = torch.distributions.Categorical(logits=self.discrete_logits)
        # Create the base continuous distribution (Normal).
        self.base_normal = torch.distributions.Normal(self.continuous_means, self.continuous_stds)
        
        # Apply transforms:
        # 1. Sigmoid maps R -> (0, 1)
        # 2. AffineTransform with scale=0.1 maps (0,1) -> (0, 0.1)
        self.transforms = [torch.distributions.SigmoidTransform(), 
                      torch.distributions.AffineTransform(
                          loc=0, scale=self.pressure_range[1])]
        self.transformed_cont_dist = torch.distributions.TransformedDistribution(self.base_normal, self.transforms)
        
        # Wrap in an Independent to treat the last dimension as the event dimension.
        self.component_dist = torch.distributions.Independent(self.transformed_cont_dist, 1)
        # Create the joint mixture distribution.
        self.joint_dist = torch.distributions.MixtureSameFamily(
            mixture_distribution=self.mixture_dist,
            component_distribution=self.component_dist
        )
        return self.joint_dist

    def log_prob(self, value):
        return self.get_distribution().log_prob(value)

    def kl(self, other):
        # Compute KL divergence between two joint distributions.
        p_probs = F.softmax(self.discrete_logits, dim=0)
        q_probs = F.softmax(other.discrete_logits, dim=0)
        kl_discrete = torch.sum(p_probs * (torch.log(p_probs + 1e-8) - torch.log(q_probs + 1e-8)))
        
        # For the continuous part, we need to compute KL on the base distributions and account for the transforms.
        # Note: There is no simple closed-form for the KL of transformed distributions, so here we
        # compute the KL of the base Normal distributions (this is an approximation).
        p_cont = torch.distributions.Independent(torch.distributions.Normal(
            self.continuous_means, self.continuous_stds), 1)
        q_cont = torch.distributions.Independent(torch.distributions.Normal(
            other.continuous_means, other.continuous_stds), 1)
        kl_components = torch.distributions.kl_divergence(p_cont, q_cont)
        kl_continuous = torch.sum(p_probs * kl_components)
        
        return kl_discrete + kl_continuous

    def to_tensors(self):
        return {'discrete_logits': self.discrete_logits,
                'continuous_means': self.continuous_means,
                'continuous_stds': self.continuous_stds,
                'discrete_values': self.discrete_values}

    @classmethod
    def from_tensors(cls, tensors):
        return cls(tensors['discrete_logits'],
                   tensors['continuous_means'],
                   tensors['continuous_stds'],
                   tensors['discrete_values'])


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
        
        #number of possible combination of body parameters
        self.discrete_combinations = list(product(self.no_chamber, self.chamber_length,self.thickness))
        self.N = len(self.discrete_combinations)
        self.D = len(self.no_chamber)
        # Initialize discrete scores (which, when multiplied by beta, form logits).
        self.scores = torch.nn.Parameter(torch.zeros((self.N,), dtype=torch.float32), requires_grad=False)
        # Continuous parameters (not updated by gradients but via update() method).
        self.continuous_means = torch.full((self.N , self.D), 0.1)  # or init mean = self.pressure_range.mean().item()
        self.continuous_stds = torch.full((self.N , self.D), 0.1) # init std = self.pressure_range.mean().item()/2
        self.target = LinearSchedule(np.log(self.N), 0, ent_decay_start, ent_decay_end)
        self.std_target = LinearSchedule(0.1, 0.005, ent_decay_start, ent_decay_end)
        self.beta = 0.0
        self.beta_min = 0.0
        self.beta_max = 5.0
        self.init_list_sample = 0
        self.baseline = 0.0               # moving average of rewards
        self.beta_b   = 0.05              # smoothing factor for baseline
        wandb.define_metric(f"design_{self.cut_off_length}/*", step_metric="train/step")

    def get_design_dist(self):
        # The discrete logits are computed as beta * scores.
        return RobotDesignDist(discrete_logits=self.beta * self.scores,
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
        while torch.abs(ent - target) > 0.01:
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
        # Get the current design distribution (an instance of RobotDesignDist).
        design_dist = self.get_design_dist()  # RobotDesignDist instance
        if abs(target - np.log(self.N)) <= 0.0001: 
            if self.init_list_sample < len(self.discrete_combinations):
                discrete_values = self.discrete_combinations[int(self.init_list_sample)]
            else:
                self.init_list_sample = 0
                discrete_values = self.discrete_combinations[int(self.init_list_sample)]
            self.init_list_sample += 1
        else:
            # ent = torch.distributions.Categorical(logits=design_dist.discrete_logits).entropy()
            # Build a categorical distribution from the discrete logits.
            disc_dist = torch.distributions.Categorical(logits=design_dist.discrete_logits)
            # Sample a discrete index (scalar tensor).
            discrete_idx = disc_dist.sample()  
            # Look up the discrete value (tuple) from your discrete_values list.
            discrete_values = self.discrete_combinations[discrete_idx.item()]
        # Create a continuous distribution for the chosen component.
        design_dist = self.get_design_dist()  # RobotDesignDist instance
        transformed_pressure_list = design_dist.get_distribution().sample()
        # Starting with the sample from the transformed distribution:
        inversed_pressure_list = transformed_pressure_list
        # Loop over the transforms in reverse order and apply the inverse:
        for transform in reversed(design_dist.transformed_cont_dist.transforms):
            inversed_pressure_list = transform.inv(inversed_pressure_list)
        return {'no_chamber': discrete_values[0],
                'chamber_length': discrete_values[1],
                'thickness': discrete_values[2],
                'pressure1': torch.clamp(inversed_pressure_list[0], min=self.pressure_range[0],max=self.pressure_range[1]),
                'pressure2': torch.clamp(inversed_pressure_list[1], min=self.pressure_range[0],max=self.pressure_range[1])} ### pressure_list is 'list' type then there is no .numpy()
    
    def update_normal_dist(self, idx, sample, reward,t,max_std=0.15,alpha = 0.2):
        """
        sample : tensor shape (2,)          # [p1, p2] sampled pressures
        reward : float
        """
        mu     = self.continuous_means[idx]     # (2,)
        sigma  = self.continuous_stds[idx]      # (2,)
        sigma = sigma.clamp(min=1e-4)          # avoid divide-by-zero
        sample = torch.as_tensor(sample, dtype=mu.dtype, device=mu.device)

        # 1. Advantage: reward minus baseline
        adv = reward - self.baseline

        # 2. Parameter deltas (element-wise)
        delta_mu    =  alpha * adv * (sample - mu)
        delta_sigma =  alpha * adv * ((sample - mu)**2 - sigma**2) / (sigma+ 1e-8)

        # 3. Apply
        self.continuous_means.data[idx] += delta_mu
        self.continuous_stds.data[idx]  += delta_sigma
        scheduled_std = self.std_target(t)
        # new_std = torch.clamp(self.continuous_stds.data[idx], min=scheduled_std,max=max_std)
        self.continuous_stds.data[idx].clamp_(min=scheduled_std, max=max_std)

    def update(self, designs, rewards, t):
        batch_reward_sums = {}
        batch_counts = {}
        batch_cont_samples = {}
        batch_disc_rewards_top = {}
        
        # Process each sample in the batch.
        for sample, reward in zip(designs, rewards):
            # Extract the discrete part as a tuple.
            d_val = (sample['no_chamber'], sample['chamber_length'], sample['thickness'])
            try:
                idx = self.discrete_combinations.index(d_val)
            except ValueError:
                raise ValueError(f"Design value {d_val} not found in {self.discrete_combinations}")
            
            # Initialize if not already.
            if idx not in batch_reward_sums and idx not in batch_disc_rewards_top:
                batch_reward_sums[idx] = 0.0
                batch_counts[idx] = 0.0
                batch_cont_samples[idx] = []
                batch_disc_rewards_top[idx] = 0.0
            # Collect continuous samples (pressures) for this index.
            cont_value = torch.tensor([
                sample['pressure1'],
                sample['pressure2']
            ], dtype=torch.float)
            batch_cont_samples[idx].append(cont_value)
            # Accumulate reward and count for this index.
            batch_reward_sums[idx] += reward
            batch_counts[idx] += 1

            if reward > batch_disc_rewards_top[idx]:
                batch_disc_rewards_top[idx] = reward  
            self.update_normal_dist(idx, cont_value, reward,t)

        for idx, top_score in batch_disc_rewards_top.items():
            self.scores.data[idx] = top_score
        # --- update moving baseline -----------------------------------------
        batch_mean_reward = torch.as_tensor(rewards, dtype=torch.float).mean().item()
        self.baseline = (1-self.beta_b)*self.baseline + self.beta_b*batch_mean_reward

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
