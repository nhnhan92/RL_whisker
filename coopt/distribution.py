import torch
import numpy as np


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
