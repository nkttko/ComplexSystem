#!/usr/bin/python
from MiningModel import *
import matplotlib.pyplot as plt

# number of instances
num_instances = 10
# number of steps
num_steps = 1
# miners
all_miners=[]

# run the model
for j in range(num_instances):
    model = MiningModel()
    for i in range(num_steps):
        model.step()

# store the results
for miners in model.num_miners_vector:
    all_miners.append(miners)

left = np.array([1,2,3,4,5,6,7,8,9,10,11])
#left = [0,1,2,3,4,5,6,7,8,9,10]
#left = [0,1]
#height = model.num_miners_vector
height = all_miners
#print(height)
#print(len(height))
#print(len(left))
plt.bar(left,height)
plt.show()
