import gym
from gym import envs
#print(envs.registry.all())
"""
env = gym.make('CartPole-v0')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)
"""
env = gym.make('MountainCar-v0')
#env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)


for i_episode in range(20):
    observation = env.reset()
    for t in range(50000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        #print(action, reward, done, observation)
        if done:
            print("Episode finished after {} timesteps. reward = {}".format(t+1, reward))
            break
