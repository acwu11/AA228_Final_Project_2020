import numpy as np
import math

from model_based import *

LEARNING_RATE = 0.9
DISCOUNT = 0.9
EPSILON = 0.3

class QAgent:
    def __init__(self):
        # state & action space representation
        self.saRep = Descritize(80, 0, 50, 0)
        nState = self.saRep.get_total_states()
        nAction = self.saRep.get_total_actions()

        # initialize the Q table
        self.Q_table = np.ones((nState, nAction))
        for a in range(9):
            self.Q_table[:, 1] = np.ones(81*51*2*2) * 5

    #Outputs vector Q(s,a) for every possible a
    def update_Q(self, obs, reward, action, next_state):
        # get the linear index from state representation
        state = self.saRep.get_state_ind(obs[0], obs[1], obs[2], obs[3])
        nex_state = self.saRep.get_state_ind(next_state[0],
            next_state[1], next_state[2], next_state[3])
    
        # update the Q table
        self.Q_table[state, action] = self.Q_table[state, action] \
            + LEARNING_RATE*(reward + DISCOUNT*max(self.Q_table[nex_state, :]) \
            - self.Q_table[state, action])

        return self.Q_table[state, :]
        
    def get_Q(self, obs):
        state = self.saRep.get_state_ind(obs[0], obs[1], obs[2], obs[3])
        print(self.Q_table[state, :])
        return self.Q_table[state, :]
        
#np.save('Q_table.npy', Q_table)
#print(Q_table[1, 1])

