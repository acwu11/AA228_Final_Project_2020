import numpy as np
import math
from representation import *

LEARNING_RATE = 0.9
DISCOUNT = 0.9
EPSILON = 0.3

#---------------------------------------------------------------------------
# Q LEARNING 
#---------------------------------------------------------------------------

class QAgent:
    def __init__(self, max_speed, min_speed, max_dist, min_dist):
        # state & action space representation
        self.saRep = Descritize(max_speed, min_speed, max_dist, min_dist)
        nState = self.saRep.NSTATE_TOT
        nAction = self.saRep.NACT_TOT

        # initialize the Q table
        self.Q_table = np.ones((nState, nAction))
        for a in range(9):
            self.Q_table[:, 1] = np.ones(nState) * 5

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

#---------------------------------------------------------------------------
# SARSA 
#---------------------------------------------------------------------------

class SARSA_agent(QAgent):
    def __init__(self, max_speed, min_speed, max_dist, min_dist):
        QAgent.__init__(self, max_speed, min_speed, max_dist, min_dist)

        # initialize state to 0 speed, 50 depth, not in reverse, no collision
        s_ind = self.saRep.get_state_ind(min_speed,max_dist,0,0)
        # initialize action to 1 throttle, no steer
        a_ind = 1
        self.lastexp = [s_ind, a_ind]

    def sarsa_update(self, r, a, sp):
        # get linear indices
        sp_ind = self.saRep.get_state_ind(sp[0], sp[1], sp[2], sp[3])
        ap_ind = np.argmax(self.Q_table[sp_ind,:])

        # update Q matrix
        if len(self.lastexp) != 0:
            s_ind = self.lastexp[0]
            a_ind = a
            self.Q_table[s_ind, a_ind] += LEARNING_RATE * (r \
                + DISCOUNT * self.Q_table[sp_ind, ap_ind] \
                - self.Q_table[s_ind, a_ind])

        # update last experience
        self.lastexp = [sp_ind, ap_ind]


#---------------------------------------------------------------------------
# SARSA LAMBDA
#---------------------------------------------------------------------------

class SLambda_agent(QAgent):
    def __init__(self, max_speed, min_speed, max_dist, min_dist, l):
        QAgent.__init__(self, max_speed, min_speed, max_dist, min_dist)
        self.nState = self.saRep.NSTATE_TOT
        self.nAction = self.saRep.NACT_TOT

        # initialize state to 0 speed, 50 depth, not in reverse, no collision and
        # action to throttle 1, steer 0
        s_ind = self.saRep.get_state_ind(min_speed,max_dist,0,0)
        a_ind = 1
        self.lastexp = [s_ind, a_ind]
        self.lam = l                                    # exponential decay factor
        self.N = np.zeros((self.nState, self.nAction))    # counts matrix
    
    def sarsa_lambda_update(self, r, a, sp):
        if len(self.lastexp) != 0:
            # update counts
            s_ind = self.lastexp[0]
            a_ind = a
            self.N[s_ind, a_ind] += 1

            # compute temporal difference update
            sp_ind = self.saRep.get_state_ind(sp[0], sp[1], sp[2], sp[3])
            ap_ind = np.argmax(self.Q_table[sp_ind,:])
            delta = r + DISCOUNT*self.Q_table[sp_ind, ap_ind]  - self.Q_table[s_ind, a_ind]

            # update counts and Q table
            for ss in range(self.nState):
                for aa in range(self.nAction):
                    self.Q_table[ss, aa] += LEARNING_RATE * delta * self.N[ss, aa]
                    self.N[ss, aa] *= DISCOUNT * self.lam

            # update last experience
            self.lastexp = [sp_ind, ap_ind]
                
        