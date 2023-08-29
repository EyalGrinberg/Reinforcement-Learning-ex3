import gym
import numpy as np

# Load environment
env = gym.make('FrozenLake-v0')

# Implement Q-Table learning algorithm
#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .8
gamma = .95
num_episodes = 2000
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
env.seed(42)
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0 # Total reward during current episode
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j += 1
        a = np.argmax([Q[s] + np.random.uniform(0, 0.1, env.action_space.n)])
        ns, r, d, info = env.step(a)
        Q[s][a] += lr * (r + gamma * np.max(Q[ns]) - Q[s][a])
        rAll += r
        if d:
            break
        else:
            s = ns        
    rList.append(rAll)

# Reports
print("Score over time: " +  str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
