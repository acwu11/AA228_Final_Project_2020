import numpy as np
import math

# Class for keeping track of the state space and action space 
# parameters, size, and indices. 
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
        self.NSTATE_TOT = self.get_total_states()
        self.NACT_TOT = 3

    # returns total size of state space
    def get_total_states(self):
        return self.nSBins * self.nDBins * self.nRev * self.nCol

    # returns total size of action space
    def get_total_actions(self):
        a = 3
        return a
   
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