import gym
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Load environment
env = gym.make('FrozenLake-v0')

# Define the neural network mapping 16x1 one hot vector to a vector of 4 Q values
# and training loss
lr = 0.01
input_size = 16
output_size = 4
class Net(nn.Module):
    def __init__(self, input_size=input_size, output_size=output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        return out
net = Net()
criterion = nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# Implement Q-Network learning algorithm
def one_hot(s, input_size=input_size):
    res = torch.FloatTensor(1, input_size).zero_()
    res[0][s] = 1
    return res


# Set learning parameters
gamma = .99
eps = 0.1
num_episodes = 4000
# create lists to contain total rewards and steps per episode
jList = []
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Network
    while j < 99:
        j += 1
        # 1. Choose an action greedily from the Q-network
        #    (run the network for current state and choose the action with the maxQ)
        Q = net(one_hot(s))
        action = torch.argmax(Q).item()

        # 2. A chance of e to perform random action
        if np.random.rand(1) < eps:
            action = env.action_space.sample()

        # 3. Get new state(mark as s1) and reward(mark as r) from environment
        s1, r, d, _ = env.step(action)

        # 4. Obtain the Q'(mark as Q1) values by feeding the new state through our network
        Q1 = net(one_hot(s1))

        # 5. Obtain maxQ' and set our target value for chosen action using the bellman equation.
        Q1_max = torch.max(Q1).item()
        Q_target = Variable(Q.data)
        Q_target[0][action] = r + gamma * Q1_max

        # 6. Train the network using target and predicted Q values (model.zero(), forward, backward, optim.step)
        Q = net(one_hot(s))
        loss = criterion(Q, Q_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rAll += r
        s = s1
        if d == True:
            #Reduce chance of random action as we train the model.
            eps = 1./((i/50) + 10)
            break
    jList.append(j)
    rList.append(rAll)

# Reports
print("Score over time: " + str(sum(rList)/num_episodes))
