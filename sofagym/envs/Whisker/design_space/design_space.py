import gym
import math as m
from itertools import product

class whiskerdesignspace():
    def __init__(self,body_length,no_chamber,chamber_length,thickness,pressure_range):
        self.no_chamber = no_chamber
        self.body_length = body_length
        self.pressure_range = pressure_range
        self.chamber_length = chamber_length
        self.thickness = thickness
        self.discrete_combinations = list(product(self.no_chamber, self.chamber_length,self.thickness))
    def design_space(self):
        design_space = {}
        for i in self.body_length:
            design_space[i] = {"no_chamber": self.no_chamber,
                               "chamber_length": self.chamber_length,
                               "thickness": self.thickness,
                                "pressure_range": self.pressure_range}
    
        return design_space
            
if __name__ == '__main__':
    import torch
    import numpy as np
    no_chamber = torch.tensor([1,2])
    pressure_range = torch.tensor([0.05,0.2])
    body_length = np.linspace(60,100,5,dtype=int)
    print(body_length)
    thickness = torch.tensor(np.linspace(2,4,5))
    chamber_length = torch.tensor(np.linspace(20,40,11,dtype=int))
    ins = whiskerdesignspace(body_length,no_chamber,chamber_length,thickness,pressure_range)
    space = ins.design_space()
    print(len(ins.discrete_combinations))
    print("Design space keys:", list(space.keys()))
    print(space[100])
    print(space[100].items())
    # for idx, val in enumerate(space[100]):
    #     print(val['no_chamber'])
    print(space[100]['no_chamber'])
    # print(pressure_range)
    