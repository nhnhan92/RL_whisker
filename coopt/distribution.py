import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily

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

class RobotDesignDistMixtureMVN(RobotDesignDist):
    def __init__(self, *args, K=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = K
        # Inner mixture logits: shape (N, K)
        self.inner_logits = nn.Parameter(torch.zeros(self.N, K))
        # Means per design per component: shape (N, K, 2)
        self.inner_means  = nn.Parameter(torch.randn(self.N, K, 2) * 0.05 + 0.15)
        # Raw lower-triangular L for each design/component: (N, K, 2, 2)
        self.inner_rawL   = nn.Parameter(torch.stack(
                                [torch.eye(2) for _ in range(self.N * K)],
                                dim=0
                             ).view(self.N, K, 2, 2))

    def get_distribution(self):
        # Outer design mix (unchanged)
        cat_design = Categorical(logits=self.beta * self.scores)

        # Build one mixture‐of‐Gaussians per design index
        inner_mixtures = []
        for d in range(self.N):
            # build covariances Σ_{d,k} = L L^T
            Ld = torch.tril(self.inner_rawL[d])           # (K,2,2)
            diag = torch.clamp(torch.diagonal(Ld, -2, -1), min=1e-3)
            Ld = Ld - torch.diag_embed(torch.diagonal(Ld, -2, -1)) \
                   + torch.diag_embed(diag)
            covs = Ld @ Ld.transpose(-2, -1)              # (K,2,2)

            mvn_comp = MultivariateNormal(
                loc=self.inner_means[d],                  # (K,2)
                covariance_matrix=covs                    # (K,2,2)
            )
            cat_comp = Categorical(logits=self.inner_logits[d])  # (K,)
            inner_mixtures.append(MixtureSameFamily(cat_comp, mvn_comp))

        # Return a wrapper that first samples a design d then inner_mixtures[d]
        return cat_design, inner_mixtures

class RobotDesignDistCorrelated(nn.Module):
    def __init__(self, discrete_logits,pressure_range, continuous_means, rawL):
        super().__init__()
        self.discrete_logits = discrete_logits
        self.continuous_means = continuous_means
        self.raw_L = rawL
        self.pressure_range = pressure_range

    def get_distribution(self):
        # 1) compute lower-triangular L with positive diag
        L = torch.tril(self.raw_L)                     # shape (N,2,2)
        # diag = torch.clamp(torch.diag(L), min=1e-3)        # shape (2,)
        diag = torch.clamp(torch.diagonal(L, dim1=1, dim2=2), min=1e-3)
        L_no_diag = torch.tril(L, diagonal=-1)  # shape (N,2,2)
        L = L_no_diag + torch.diag_embed(diag) 

        cov = L @ L.transpose(-1, -2)                  # Σ = L Lᵀ, shape (N,2,2)
        mean = self.continuous_means                   # (N,2)

        self.mvn = MultivariateNormal(loc=mean, covariance_matrix=cov)

        self.transforms = [torch.distributions.SigmoidTransform(), 
                      torch.distributions.AffineTransform(
                          loc=0, scale=(self.pressure_range[1]-self.pressure_range[0]))]
        # 4) wrap with our transforms so final support is [pmin,pmax]^2
        self.transformed_cont_dist = torch.distributions.TransformedDistribution(self.mvn, self.transforms)
        self.cat = Categorical(logits=self.discrete_logits)
        return MixtureSameFamily(mixture_distribution=self.cat,
            component_distribution=self.transformed_cont_dist)

