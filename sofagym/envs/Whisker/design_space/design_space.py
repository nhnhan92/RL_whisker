import gym
import math as m

no_chamber = 3
no_regulator = 3
body_length = [100, 80, 60]
class whiskerdesignspace():
    def __init__(self):
        self.pressure_range = gym.spaces.Box(low=0.0001, high=0.001, shape=(1,), dtype='float32')
        self.alldesign = self.design_space()
        # self.body_length = gym.spaces.Box(low=80, high=100, shape=(1,), dtype='float32')
        

    def design_space(self):
        design = []
        for i in range(len(body_length)):
            for j in range(1,no_chamber+1):
                sub_array = [body_length[i],j,0,0,0]
                for k in range (2,j+2):
                    sub_array[k] = self.pressure_range.sample().item()
                design.append(sub_array)
    
        return design
            
