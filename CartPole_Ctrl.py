import gym
import msvcrt

LEFT = 0
RIGHT = 1

arrow_keys = {
b'd': RIGHT,
b'a': LEFT
}    

#env = gym.make('CartPole-v0')
env = gym.make('MountainCar-v0')
print(env.action_space)
print(env.observation_space)
observation = env.reset()
env.render()

while True:
    key = msvcrt.getch()
    if key not in arrow_keys.keys():
        print('Game Aborted!')
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()
    print('State: ', state, 'Action: ', action, 'Reward: ', reward, 'Info: ', info)

    if done:
        observation = env.reset()
        print('Finished with reward: ', reward)
        break
      


