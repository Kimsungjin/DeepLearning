import gym

env = gym.make('CartPole-v0')
env.reset()
episode = 0
reward_sum = 0
while episode < 10:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    print(observation, reward, done)
    reward_sum += reward
    if done:
        episode += 1
        print('Reward for this episode was:', reward_sum)
        reward_sum = 0
        env.reset()



