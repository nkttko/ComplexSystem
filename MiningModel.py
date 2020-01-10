from mesa import Agent, Model
from mesa.time import RandomActivation
import numpy as np
import random
import sys

# ----Global Variables-----

# number of miners in the network
num_miners = 100
# number of pools in the network
num_pools = 100
# rewards per a mining
default_rewards = 10
# total hash rate (GH/s)
#hashrate = 1
# costs required for a mining
default_costs = 1 / 10000
# fee required for a mining
default_fee = 1 / 100000
# risk preference setting
risk_pref = [
        [1, 1],   #neutral
        [1.2, 1], #risk-seeking
        [1, 1.2]  #risk-adverse
    ] 

# matrix of miners and pools ( a[i][j]=1 if a miner i selects a pool j, otherwise a[i][j]=0 )
network_matrix = np.zeros((num_miners, num_pools))
network_matrix = np.insert(network_matrix, 0, 1, axis=1)
# matrix of expected utility
expected_utility_matrix = np.zeros((num_miners, num_pools+1))
# vector of fee of pools
fee_vector = np.full((1, num_pools+1), default_fee)
# vector of the number of miners in a pool
num_miners_vector = np.zeros((1, num_pools+1))
num_miners_vector[0][0] = 0

# ----Agents----

class MinerAgent(Agent):

    global network_matrix
    global expected_utility_matrix

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.id = unique_id

    def step(self):
        # select risk preference randomly
        self.miner_risk_pref = random.choice([0,1,2])
        # vector of costs of mining
        self.miner_costs = default_costs
        # compute expected return for all pools, select the highest
        for i in range(num_pools+1):
            # compute expected return
            if num_miners_vector[0][i] == 0:
                self.miner_prob = 1 / num_miners
            else: 
                self.miner_prob = 1 / num_miners * num_miners_vector[0][i]
            if num_miners_vector[0][i] == 0:
                self.miner_rewards = default_rewards
            else:
                self.miner_rewards = default_rewards / num_miners_vector[0][i]
            # compute expected utility & update expected utility matrix
            expected_utility_matrix[self.id][i] = \
            risk_pref[self.miner_risk_pref][0] * risk_pref[self.miner_risk_pref][1] * \
            (self.miner_prob * self.miner_rewards - self.miner_costs - fee_vector[0][i])
        # select the highest
        temp = expected_utility_matrix[self.id]
        index = [i for i, x in enumerate(temp) if x == max(temp)]
        if len(index) > 1:
            index = random.choice(list(enumerate(temp)))[0]
        # update network matrix
        network_matrix[self.id].fill(0)
        network_matrix[self.id][index] = 1

# ----Models----

class MiningModel(Model):

    global num_miners_vector
        
    def __init__(self):
        # create miner agents
        self.schedule = RandomActivation(self)
        for i in range(num_miners):
            ma = MinerAgent(i, self)
            self.schedule.add(ma)

    def step(self):
        # each miner chooses the best strategy
        self.schedule.step()
        # update the number of miners in a pool
        num_miners_vector = np.sum(network_matrix, axis=0)
        num_miners_vector[0] = 1
        # store data
        self.network_matrix = network_matrix
        self.num_miners_vector = num_miners_vector
