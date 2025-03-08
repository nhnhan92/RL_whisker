import gym
import math as m
from itertools import product

class whiskerdesignspace():
    def __init__(self,body_length,no_chamber,pressure_range):
        self.no_chamber = no_chamber
        self.body_length = body_length
        self.pressure_range = pressure_range
        # All possible combinations between body_length and no_chamber as tuples.
        self.discrete_combinations = list(product(self.body_length, self.no_chamber))
        self.num_combinations = len(self.discrete_combinations)
    def design_space(self):
        design = {"body_length": self.body_length,
                  "no_chamber": self.no_chamber,
                  "pressure_range": self.pressure_range}
    
        return design
            
if __name__ == '__main__':
    no_chamber = [1,2,3]
    pressure_range = [0,0.01]
    body_length = [100, 80, 60]
    ins = whiskerdesignspace(body_length,no_chamber,pressure_range)
    print(ins)