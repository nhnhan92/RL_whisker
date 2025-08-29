import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product
import gym
import wandb
import math as m
from coopt.distribution import RobotDesignDist,RobotDesignDistMixtureMVN,RobotDesignDistCorrelated,HybridCorrelatedRobotDesignDist
from coopt.utils import RunningMeanStd, RunningMeanStd_limited
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def to_phys(z, pmin, prng):
    # z: (...,2)
    return pmin + torch.sigmoid(z) * prng

def to_latent(p, pmin, prng, eps=1e-6):
    # p: (...,2) in [pmin, pmin+prng]
    y = torch.clamp((p - pmin) / prng, eps, 1.0 - eps)
    return torch.log(y) - torch.log1p(-y)  # logit
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
        self.ent_decay_end = ent_decay_end
        self.ent_decay_start = ent_decay_start
        #number of possible combination of body parameters
        self.discrete_combinations = list(product(self.no_chamber, self.chamber_length,self.thickness))
        # Build mask
        self.active_mask = torch.ones(len(self.discrete_combinations), 2)
        for i,(no_chamber, _, _) in enumerate(self.discrete_combinations):
            if no_chamber == 1:
                self.active_mask[i,1] = 0  # second pressure inactive
        self.N = len(self.discrete_combinations)
        self.reward_rms = [RunningMeanStd() for _ in range(self.N)]
        self.tracking_scores = [RunningMeanStd_limited(window_size=40) for _ in range(self.N)]
        self.D = len(self.no_chamber)
        K = 3 ## Number of inner normal distribution
        # Initialize discrete scores (which, when multiplied by beta, form logits).
        self.scores = torch.nn.Parameter(torch.zeros((self.N,), dtype=torch.float32), requires_grad=False)
        # Continuous parameters (not updated by gradients but via update() method).
        continuous_means_np = np.random.uniform(0.01, 0.05, size=(self.N, self.D))
        continuous_means_np= torch.tensor(continuous_means_np, dtype=torch.float32)
        self.continuous_means = to_latent(continuous_means_np,self.pressure_range[0], self.pressure_range[1])
        # self.continuous_means = to_latent(torch.zeros(self.N,self.D) + 0.01,self.pressure_range[0], self.pressure_range[1])

        # Means: latent-centered (0.1) with small jitter on active dims
        # mu_lat = torch.zeros(self.N, self.D, dtype=torch.float32)
        # mu_lat += 0.2 * torch.randn(self.N, self.D) * self.active_mask  # jitter only active dims

        # # For inactive dims, optionally pin to latent(pmin)
        # pin_lat = to_latent(torch.tensor(self.pressure_range[0]), self.pressure_range[0], self.pressure_range[1]).item()
        # mu_lat[self.active_mask == 0] = pin_lat
        # self.continuous_means = mu_lat

        l0 = torch.eye(2) * 2
        self.continuous_stds = nn.Parameter(torch.stack([l0 for _ in range(self.N)], dim=0),requires_grad=False) 
        self.target = LinearSchedule(np.log(self.N), 0, ent_decay_start, ent_decay_end)
        self.std_target_min = LinearSchedule(0.1, 0.001, ent_decay_start, ent_decay_end)  ## in latent space
        self.std_target_max = LinearSchedule(1.2, 0.01, ent_decay_start, ent_decay_end)  ## in latent space
        self.beta = 0.0
        self.beta_min = 0.0
        self.beta_max = 5.0
        self.init_list_sample = 0
        self.pressure_grid_idx = 0
        self.baseline = torch.zeros(len(self.discrete_combinations), dtype=torch.float)
        self.scores_baseline = torch.zeros(len(self.discrete_combinations), dtype=torch.float)
        self.beta_b   = 0.15           # smoothing factor for baseline
        self.beta_b_target = LinearSchedule(0.1, 0.05, ent_decay_start, ent_decay_end)
        self.alpha_target = LinearSchedule(0.2, 0.05, ent_decay_start, ent_decay_end)
        self.alpha_score = LinearSchedule(0.15, 0.05, ent_decay_start, ent_decay_end)
        
        wandb.define_metric(f"design_{self.cut_off_length}/*", step_metric="train/step")

    def get_design_dist(self,type = 'covariance'):
        # The discrete logits are computed as beta * scores.
        if type == 'single_variance':
            return RobotDesignDist(discrete_logits=self.beta * self.scores,
                                pressure_range = self.pressure_range,
                                continuous_means=self.continuous_means,
                                rawL=self.continuous_stds,
                                discrete_values=self.discrete_combinations)
        elif type == 'covariance':
            return HybridCorrelatedRobotDesignDist(discrete_logits=self.beta * self.scores,
                                pressure_range = self.pressure_range,
                                continuous_means=self.continuous_means,
                                rawL=self.continuous_stds,
                                active_mask=self.active_mask)

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
        robot_dist = self.get_design_dist()
        cat, mvn = robot_dist.get_distribution()  

        if abs(target - np.log(self.N)) <= 1e-4: 
            discrete_idx = self.init_list_sample
            self.init_list_sample = (discrete_idx + 1) % len(self.discrete_combinations)
        else:
            discrete_idx = int(cat.sample().item())
        # Discrete values
        discrete_idx = 21
        discrete_values = self.discrete_combinations[discrete_idx]
        mask = self.active_mask[discrete_idx] 
        # Sample raw 2-D (even if one dim inactive)
        raw_vec_all = mvn.sample()               # shape (N,2)
        pmin = float(self.pressure_range[0])
        prng = float(self.pressure_range[1] - self.pressure_range[0])
        if t <=self.ent_decay_end:
            raw_z = raw_vec_all[discrete_idx]        # raw sample (2,)
            # Raw → physical transform (sigmoid + affine)            
        else:
            raw_z = self.continuous_means[discrete_idx]
        # Masking
        p = to_phys(raw_z, pmin, prng)
        pressures = p * mask + pmin * (1 - mask)
        pressures = torch.clamp(pressures,
                        min=self.pressure_range[0],
                        max=self.pressure_range[1])
        pressures = torch.tensor([0.01159,0])
        return {'no_chamber': discrete_values[0],
                'chamber_length': discrete_values[1],
                'thickness': discrete_values[2],
                'pressure1': pressures[0],
                'pressure2': pressures[1]}
    
    def robust_cholesky_2x2(self,Sigma, 
                            min_eig=1e-3,   # floor each eigenvalue
                            jitter_start=1e-6,  # initial diag bump
                            jitter_mult=10,     # ramp factor
                            max_tries=5):
        """
        Clamp eigenvalues, then try Cholesky with increasing jitter until it succeeds.
        Sigma: torch.Tensor shape (2,2), assumed symmetric.
        Returns L so that L @ L.T ~= Sigma_pd.
        """
        # 1) Symmetrize & floor eigenvalues
        Sigma = 0.5 * (Sigma + Sigma.T)
        vals, vecs = torch.linalg.eigh(Sigma)
        vals = torch.clamp(vals, min=min_eig)
        Sigma_pd = vecs @ torch.diag(vals) @ vecs.T

        # 2) Try Cholesky, adding jitter on failure
        jitter = jitter_start
        for _ in range(max_tries):
            try:
                return torch.linalg.cholesky(Sigma_pd)
            except RuntimeError:
                # Bump the diagonal by jitter
                Sigma_pd = Sigma_pd + torch.eye(2, device=Sigma_pd.device) * jitter
                jitter *= jitter_mult

        # 3) As a last‐ditch fallback, enforce positive det
        a, b = Sigma_pd[0,0], Sigma_pd[1,1]
        c    = Sigma_pd[0,1]
        # ensure a>0, b>0 already by min_eig; now force a*b - c^2 > 0
        det = a*b - c*c
        if det <= min_eig:
            # reduce off-diagonal
            max_off = torch.sqrt(a*b) * 0.999
            c = torch.clamp(c, -max_off, max_off)
            Sigma_pd[0,1] = Sigma_pd[1,0] = c
        # final Cholesky
        return torch.linalg.cholesky(0.5 * (Sigma_pd + Sigma_pd.T))

    def update_normal_dist(self, idx, sample, reward,t,z_clip = 5.0):
        """
        sample : tensor shape (2,)          # [p1, p2] sampled pressures
        reward : float
        """
        ## Compute normalized score
        self.tracking_scores[idx].update([reward])
        if t <= self.ent_decay_start and reward > self.scores.data[idx]:
            self.scores.data[idx] = reward
        elif self.ent_decay_start < t < self.ent_decay_end:
            self.tracking_scores[idx].set_window_size(10)
            raw_score = reward - self.tracking_scores[idx].mean
            denom = max(self.tracking_scores[idx].std,0.1) 
            self.scores.data[idx] = self.tracking_scores[idx].mean
        
        # Compute normalized advantage
        raw_adv = reward - self.baseline[idx]
        denom = max(self.reward_rms[idx].std,0.1)   # or 1e-2
        adv   = raw_adv/ denom
        # adv   = raw_adv
        alpha = self.alpha_target(t)
        # alpha = 0.15
        # reconstruct Σ = L Lᵀ
        Ld    = torch.tril(self.continuous_stds[idx])  
        diag = torch.clamp(torch.diag(Ld), min=1e-3)   # → shape (N,2)
        L_no_diag = torch.tril(Ld, diagonal=-1)              # strictly lower triangle
        Ld = L_no_diag + torch.diag_embed(diag) 
        sigma = Ld @ Ld.transpose(-1, -2)    # (2,2) 
        # Masking
        mask = self.active_mask[idx]
        mu     = self.continuous_means[idx]     # (2,)
        sample = torch.as_tensor(sample, dtype=mu.dtype, device=mu.device)
        z = to_latent(sample * mask,self.pressure_range[0],self.pressure_range[1])
        sample = mask * z + (1 - mask) * mu
        
        
        # if t <= self.ent_decay_start:
        #     pass
            # self.continuous_means.data[idx] = torch.clamp(mu_new,min=-z_clip,max = z_clip)
        
        if self.ent_decay_start < t <= self.ent_decay_end:
            delta = sample - mu  # inactive dims set to zero difference
            mu_new = mu + alpha * adv * delta
            self.continuous_means.data[idx] = torch.clamp(mu_new,min=-z_clip,max = z_clip)
            # 5) ΔΣ = α A [ (p-μ)(p-μ)ᵀ - Σ ]
            outer = delta.unsqueeze(1) @ delta.unsqueeze(0)  # (2,2)
            # Zero out rows/cols where mask=0 to keep them inert
            mm = mask.view(2,1) * mask.view(1,2)
            outer = outer * mm
            sigma_new = sigma + alpha * adv * (outer - sigma* mm)
            σ_min = self.std_target_min(t)   
            σ_max = self.std_target_max(t) 
            #Symmetrize + clamp eigenvalues
            sigma_new = 0.5 * (sigma_new + sigma_new.T)
            eigvals, eigvecs = torch.linalg.eigh(sigma_new)
            eigvals = torch.clamp(eigvals, min=σ_min**2, max=σ_max**2)
            sigma_clamped = eigvecs @ torch.diag(eigvals) @ eigvecs.T
            self.eps_std = 1e-4
            # Re-impose inactive structure (tiny variance + zero cross terms)
            for d in range(2):
                if mask[d] == 0:
                    sigma_clamped[d, :] = 0.0
                    sigma_clamped[:, d] = 0.0
                    sigma_clamped[d, d] = self.eps_std**2
            
            # val = self.std_target_max(t)
            # sigma_target_std = torch.tensor(val, device=sigma_clamped.device, dtype=sigma_clamped.dtype)

            # Sigma_target = torch.diag(torch.full((2,), sigma_target_std**2, device=sigma_clamped.device))

            # L_new = self.robust_cholesky_2x2(Sigma_target)
            # 6) recompute L so that Sigma_clamped = L L^T
            L_new = self.robust_cholesky_2x2(sigma_clamped)

            self.continuous_stds.data[idx] = L_new
            
        # Update running mean/std with the new reward
        self.reward_rms[idx].update([reward])
        
        ## Update score for discrete???????????????
        # beta_dist_scores = 0.05
        # self.scores.data[idx] += adv * beta_dist_scores 
        # Moving baseline per design class
        # beta_b = self.beta_b_target(t)
        beta_b = 0.05
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
            
            # Update continuous distribution for this idx using top-k only
            # if t <= self.ent_decay_start:
            for c_sample, r in zip(top_cont, top_rewards):
                # Update discrete score (mean of top-k rewards)
            
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
        xs   = list(range(len(self.discrete_combinations)))
        if isinstance(self.scores, torch.Tensor):
            ys = self.scores.detach().cpu().numpy().tolist()  
        fig2, ax2 = plt.subplots(figsize=(11, 4))
        ax2.plot(xs, ys, linestyle='-', marker='s')  # line + square markers
        ax2.set_facecolor("none")
        fig2.patch.set_alpha(0)
        for spine in ("top", "right"):
            ax2.spines[spine].set_visible(False)

        ax2.set_xlabel("design idx")
        ax2.set_ylabel("score")
        ax2.set_title(f"Score per idx  (body_length={self.cut_off_length})")
        wandb.log({
            f"{self.cut_off_length}/score_line_{self.cut_off_length}": wandb.Image(fig2)
        })
        plt.close(fig2)
    
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
