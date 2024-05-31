import torch
import numpy as np
import matplotlib.pyplot as plt
from .design_space.design_space import whiskerdesignspace
import gym
# Custom categorical distribution using dl.CatDist

class CustomCategoricalDistribution:
    def __init__(self, logits):
        self.dist = torch.distributions.Categorical(logits=logits)   
        self.initial_beta = 1.0  # Initial inverse temperature

    # def get_design_space(self):
    #     self._alldesign = whiskerdesignspace.design_space()
    #     return self._alldesign

    def sample(self):
        return self.dist.sample()

def gibbs_distribution_for_categories(logits, beta):
    custom_dist = CustomCategoricalDistribution(logits * beta)
    return custom_dist
    # Function to update Gibbs distribution for categories over time and sample
def update_gibbs_distribution_for_categories(design_space, initial_logits, initial_beta):
    # global sample
    categories = list(range(1,10))
    # plt.plot(categories, gibbs_distribution_for_categories(initial_logits, initial_beta).dist.probs.numpy(), label='Initial Distribution')

    updated_logits = initial_logits + torch.tensor(np.random.rand(initial_logits.shape[0]))  # Update logits with random values
    # updated_beta = initial_beta + update_beta
    current_distribution = gibbs_distribution_for_categories(updated_logits, initial_beta)  # Calculate current distribution
    # plt.plot(categories, current_distribution.dist.probs.numpy(), linestyle='--', label=f'Step = {step}')
    
    # Sample from the current distribution
    sample = current_distribution.sample()
    # print(f"Sampling Category {design_space[sample]}")

    # plt.title('Gibbs Distribution for Categories Over Time')
    # plt.xlabel('Categories')
    # plt.ylabel('Probability')
    # plt.legend()
    # plt.show()
    return sample
    

# class sampling:
#     def __init__(self):


# # Update and plot Gibbs distribution for categories over time
# print(CustomCategoricalDistribution.initial_logits)
if __name__ == '__main__':
    design_space = whiskerdesignspace.design_space() # Categories from 1 to 10

    initial_logits = torch.ones(len(design_space),dtype=torch.float)
    num_steps = 5  # Number of update steps
    initial_beta = 1.0  # Initial inverse temperature
    sampled_design = update_gibbs_distribution_for_categories(design_space = design_space,
                                                                initial_logits = initial_logits, 
                                                                initial_beta = initial_beta)
    print(design_space[sampled_design.item()])