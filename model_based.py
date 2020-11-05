# model based reinforcement learning
import numpy as np
import math

#------------------------------------------------------------
# MAXIMUM LIKELIHOOD 
# -----------------------------------------------------------
class MaxLProblem:

    # initialize maximum likelihood MDP problem
    def __init__(self, nState, nAction, discount):
        self.nS = nState
        self.nA = nAction
        self.g = discount

        # count, transition, and rewards storage
        self.N = np.zeros((nAction, nState, nState))   # nA layers of nS x nS
        self.T = np.zeros((nAction, nState, nState))   # nA layers of nS x nS
        self.R = np.zeros(nState, nAction)             # nS x nA
        self.U = np.zeros(nState)
    
    # updates the count and rewards matrices given
    # indices of current state s, action taken a, 
    # next state sp, and the reward received r.
    def update_mats(self, s_ind, a_ind, r, sp_ind):
        self.N[a_ind, s_ind, sp_ind] += 1
        self.R[s_ind, a_ind] += r

    # performs lookahead given current state and
    # and action indices
    def lookahead(self, s_ind, a_ind):
        # total count of N(s,a) over all s'
        n = sum(self.N[a_ind,s_ind,:])
        if n == 0:
            return 0
        else:
            r = self.R[s,a] / n
            self.T[a, s, ]
        




