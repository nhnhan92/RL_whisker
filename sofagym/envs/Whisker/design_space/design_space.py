import gym
import math as m

no_chamber = 3
no_regulator = 3
body_length = [100, 90, 80]
class whiskerdesignspace():
    def __init__(self):
        self.alldesign = self.design_space()
        # self.body_length = gym.spaces.Box(low=80, high=100, shape=(1,), dtype='float32')
        self.pressure_range = gym.spaces.Box(low=0.1, high=1, shape=(1,), dtype='float32')


    def design_space():
        design = []
        for i in range(len(body_length)):
            for j in range(1,no_chamber+1):
                sub_array = [body_length[i],j,0,0,0]
                for k in range (2,j+2):
                    sub_array[k] = k-1
                design.append(sub_array)

        return design
            
if __name__ == '__main__':
    design_space = gym.spaces.Discrete(n=9)
    opt = DiscreteDesignOptimizer(design_space, 0, 100)
    for t in range(100):
        designs = torch.stack([opt.sample() for _ in range(100)])
        rewards = 10 * torch.from_numpy(np.random.rand(100)).float()
        opt.update(designs, rewards, t)
        opt.log(t)
        if t % 10 == 0:
            state = opt.state_dict()
            opt = DiscreteDesignOptimizer(design_space, 0, 100)
            opt.load_state_dict(state)