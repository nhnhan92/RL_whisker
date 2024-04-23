import torch
import numpy as np
import matplotlib.pyplot as plt

# Custom categorical distribution using dl.CatDist
class CustomCategoricalDistribution:
    def __init__(self, logits):
        self.dist = torch.distributions.Categorical(logits=logits)

    def sample(self):
        return self.dist.sample()
    
    
def gibbs_distribution_for_categories(categories, logits, beta):
    custom_dist = CustomCategoricalDistribution(logits * beta)
    return custom_dist


# Function to update Gibbs distribution for categories over time and sample
def update_gibbs_distribution_for_categories(categories, initial_logits, initial_beta, update_beta):
    plt.plot(categories, gibbs_distribution_for_categories(categories, initial_logits, initial_beta).dist.probs.numpy(), label='Initial Distribution')

    updated_logits = initial_logits + torch.tensor(np.random.rand(initial_logits.shape[0]))  # Update logits with random values
    updated_beta = initial_beta + update_beta
    
    current_distribution = gibbs_distribution_for_categories(categories, updated_logits, updated_beta)  # Calculate current distribution
    plt.plot(categories, current_distribution.dist.probs.numpy(), linestyle='--', label=f'Beta = {updated_beta}')
    
    # Sample from the current distribution
    sample = current_distribution.sample()
    print(f"Sample at step: Category {categories[sample]}")

    plt.title('Gibbs Distribution for Categories Over Time')
    plt.xlabel('Categories')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()
    return sample
categories = np.array([80,90,100]) # Categories from 1 to 10
initial_logits = torch.ones(3,dtype=torch.float)
initial_beta = 1.0  # Initial inverse temperature

# Update and plot Gibbs distribution for categories over time
update_gibbs_distribution_for_categories(categories, initial_logits, initial_beta,update_beta=0)
