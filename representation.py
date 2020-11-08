import numpy as np
import math

# Class for keeping track of the state space and action space 
# parameters, size, and indices. 
class Descritize:
    def __init__(self, max_speed, min_speed, Sbins, max_dist, min_dist, Dbins, max_distside, min_distside, DSbins, nAction):
        self.max_s = max_speed
        self.max_d = max_dist
        self.max_ds = max_distside
        
        self.min_s = min_speed
        self.min_d = min_dist
        self.min_ds = min_distside
        
        self.SBinSize = Sbins
        self.DBinSize = Dbins
        self.DSBinSize = DSbins
        
        self.nSBins = int((max_speed - min_speed)/Sbins + 1)
        self.nDBins = int((max_dist - min_dist)/Dbins + 1)
        self.nDRBins = int((max_distside - min_distside)/DSbins + 1)
        self.nDLBins = int((max_distside - min_distside)/DSbins + 1)

        self.nCol = 2                              # collision: 0 or 1
        
        self.NSTATE_TOT = self.get_total_states()
        self.NACT_TOT = nAction

    # returns total size of state space
    def get_total_states(self):
        return int(self.nSBins * self.nDBins * self.nDRBins * self.nDLBins)

    # returns total size of action space
    def get_total_actions(self):
        a = 5
        return a
   
    # gets linear state index for given car state: 
    # speed, depth, reverse (or not), and collision (or not)
    def get_state_ind(self, speed, depth, depthR, depthL):
        s = math.floor(speed/self.SBinSize)
        if s > self.nSBins-1: 
            s = self.nSBins-1

        d = math.floor(depth/self.DBinSize)
        if d > self.nDBins-1:
            d = self.nDBins-1
            
        dR = math.floor(depthR/self.DSBinSize)
        if dR > self.nDRBins-1:
            dR = self.nDRBins-1
            
        dL = math.floor(depthL/self.DSBinSize)
        if dL > self.nDLBins-1:
            dL = self.nDLBins-1
        
        return np.ravel_multi_index((s, d, dR, dL), 
            (self.nSBins, self.nDBins, self.nDRBins, self.nDLBins))