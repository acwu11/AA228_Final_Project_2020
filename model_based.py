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

    # returns total size of action space
    def get_total_actions(self):
        return 3
   
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
class MaxLAgent:

    # initialize maximum likelihood MDP problem
    def __init__(self, max_speed, min_speed, max_dist, min_dist, discount):
        # state space representation
        self.saRep = Descritize(max_speed, min_speed, max_dist, min_dist)
        nState = self.saRep.get_total_states()
        nAction = self.saRep.get_total_actions()

        # params
        self.nS = nState
        self.nA = nAction
        self.g = discount

        # count, transition, and rewards storage
        self.N = np.zeros((nAction, nState, nState))   # counts: nA layers of nS x nS
        self.T = np.zeros((nAction, nState, nState))   # transition: nA layers of nS x nS
        self.r = np.zeros(nState, nAction)             # reward (sum only): nS x nA
        self.R = np.zeros(nState, nAction)             # reward / n: nS x nA
        self.U = np.zeros(nState, nAction)             # utility: nS x nA
        

    # updates the count, transition, and rewards 
    # matrices given current state observations curr_obs, action 
    # taken a, next state observations next_obs, and the reward received r.
    def update_mats(self, curr_obs, next_obs, a, r):
        # set up matrix indices
        s_ind = self.saRep.get_state_ind(curr_obs[0], curr_obs[1], curr_obs[2], curr_obs[3])
        sp_ind = self.saRep.get_state_ind(next_obs[0], next_obs[1], next_obs[2], next_obs[3])
        a_ind = a

        # counts
        self.N[a_ind, s_ind, sp_ind] += 1
        n = sum(self.N[a_ind, s_ind, :])
        
        #  reward and transition 
        if n == 0:
            self.R[s_ind, a_ind] = 0
            self.T[a_ind, s_ind,:] = 0
        else:
            self.R[s_ind, a_ind] = self.r[s_ind, a_ind] / n
            self.T[a_ind, s_ind,:] = self.N[a_ind, s_ind, :] / n

    # performs the lookahead equation and returns utility
    def lookahead(self, curr_obs, a):
        # set up matrix indices
        s_ind = self.saRep.get_state_ind(curr_obs[0], curr_obs[1], curr_obs[2], curr_obs[3])
        a_ind = a





    
    


        




