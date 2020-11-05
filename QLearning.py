import numpy as np
import math
from model_based import *

LEARNING_RATE = 0.9
DISCOUNT = 0.9
EPSILON = 0.3

class QAgent:
    def __init__(self):
        self.Q_table = np.ones((81*51*2*2, 3))
        for a in range(9):
            self.Q_table[:, 1] = np.ones(81*51*2*2) * 5

        self.StateRep = Descritize(80, 0, 50, 0)
        
    #Outputs vector Q(s,a) for every possible a
    def update_Q(self, obs, reward, action, next_state):
        # get the linear index from state representation
        self.state = self.StateRep.get_state_ind(obs[0], obs[1], obs[2], obs[3])
        self.nex_state = self.StateRep.get_state_ind(next_state[0],
            next_state[1], next_state[2], next_state[3])
    
        # update the Q table
        self.Q_table[self.state, action] = self.Q_table[self.state, action] + 
            LEARNING_RATE*(reward + DISCOUNT*max(self.Q_table[self.nex_state, :]) 
            - self.Q_table[self.state, action])
            
        return self.Q_table[self.state, :]
        
    def get_Q(self, obs):
        self.speed = math.floor(obs[0])
        if self.speed > 80:
            self.speed = 80
            
        self.dist = math.floor(obs[1])
        if self.dist > 50:
            self.dist = 50
            
        self.rev = obs[2]
        self.col = obs[3]
        if self.col > 1:
            self.col = 1
        self.state = np.ravel_multi_index((self.speed, self.dist, self.rev, self.col), (81,51,2,2))
        print(self.Q_table[self.state, :])
        return self.Q_table[self.state, :]
        
#np.save('Q_table.npy', Q_table)
#print(Q_table[1, 1])

