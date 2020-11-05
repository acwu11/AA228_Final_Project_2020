# model based reinforcement learning
import numpy as np
import math

# -----------------------------------------------------------
# DESCRITIZE STATE AND ACTIONS
# -----------------------------------------------------------
class Descritize:
    def __init__(self, max_speed, min_speed, max_dist, min_dist):
        self.max_s = max_speed
        self.max_d = max_dist
        self.min_s = min_speed
        self.min_d = min_dist
        self.nSBins = max_speed - min_speed + 1
        self.nDBins = max_dist - min_dist + 1
        self.nRev = 2                              # reverse: 0 or 1
        self.nCol = 2                              # collision: 0 or 1

    # returns total size of state space
    def get_total_states(self):
        return self.nSBins * self.nDBins * self.nRev * self.nCol
   
    # gets linear state index for given car state: 
    # speed, depth, reverse (or not), and collision (or not)
    def get_state_ind(self, speed, depth, rev, col):
        s = math.floor(speed)
        if s > self.max_s: 
            s = self.max_s

        d = math.floor(depth)
        if d > self.max_d:
            d = self.max_d
        
        if col > 1:
            col = 1
        
        return np.ravel_multi_index((s, d, rev, col), 
            (self.nSBins, self.nDBins, self.nRev, self.nCol))

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
        self.N = np.zeros((nAction, nState, nState))   # counts nA layers of nS x nS
        self.T = np.zeros((nAction, nState, nState))   # transition nA layers of nS x nS
        self.r = np.zeros(nState, nAction)             # unnormalized reward nS x nA 
        self.R = np.zeros(nState, nAction)             # reward MLE nS x nA
        # self.U = np.zeros(nState)
    
    # updates the count, transition, and rewards 
    # matrices given indices of current state s, action 
    # taken a, next state sp, and the reward received r.
    def update_mats(self, s_ind, a_ind, r, sp_ind):
        # counts
        self.N[a_ind, s_ind, sp_ind] += 1
        n = sum(self.N[a_ind, s_ind, :])
        
        # full update
        if n == 0:
            self.R[s_ind, a_ind] = 0
            self.T[a_ind, s_ind,:] = 0
        else:
            self.R[s_ind, a_ind] = self.r[s_ind, a_ind] / n
            self.T[a_ind, s_ind,:] = self.N[a_ind, s_ind, :] / n





    
    


        




