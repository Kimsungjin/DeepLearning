import gym
import numpy as np
import matplotlib.pyplot as plt

#env = gym.make('CartPole-v0')
env = gym.make('FrozenLake-v0')

print(env.observation_space, env.action_space)

Q = np.zeros([env.observation_space.n, env.action_space.n])

lr = .85
dis = .99
episode = 2000

rList = []
for i in range(episode):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i+1))
        new_state, reward, done, _ = env.step(action)
        Q[state, action] = (1-lr) * Q[state,action] + lr * (reward + dis * np.max(Q[new_state, :]))

        rAll += reward
        state = new_state

    rList.append(rAll)

print('Score over time: ' + str(sum(rList)/episode))
print('Final Q-Table Value')
#Print(Q)
plt.bar(range(len(rList)), rList, color='blue')
plt.show()



    
