import tensorflow as tf
import numpy as np
import dqn 
import random
from collections import deque
import gym

env = gym.make('MountainCar-v0')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = 0.9
REPLAY_MEMORY = 50000

def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))        

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    return mainDQN.update(x_stack, y_stack)

def get_copy_var_ops(*, dest_scope_name='target', src_scope_name='main'):
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = dest_scope_name)

    for src_vars, dest_vars in zip(src_vars, dest_vars):
        op_holder.append(dest_vars.assign(src_vars.value()))

    return op_holder

def bot_play(mainDQN):
    sList = []
    for i in range(20):
        reward_sum=0
        s = env.reset()
        while True:
            env.render()
            a = np.argmax(mainDQN.predict(s))
            s, reward, done, _ = env.step(a)
            reward_sum += reward
            if done:
                sList.append(reward_sum)
                print('{} Total Score: {}'.format(i, reward_sum))
                break

    print('everage reward = {}'.format( np.mean(sList) ))
    

def main():
 
    max_episode = 5000
    max_try = 500
    replay_buff = deque()

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, name='main')
        targetDQN = dqn.DQN(sess, input_size, output_size, name='target')

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        copy_ops = get_copy_var_ops(dest_scope_name='target', src_scope_name='main')
        sess.run(copy_ops)

        success_count = 0
        sList = []

        for episode in range(max_episode):
            e = 1. / ((episode / 10) +1)
            done = False
            state = env.reset()

            for each_try in range(max_try):
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                next_state, reward, done, _ = env.step(action)
                if done:
                    reward = 1000

                replay_buff.append((state, action, reward, next_state, done))
                if len(replay_buff) > REPLAY_MEMORY:
                    replay_buff.popleft()

                state = next_state

                if done:
                    success_count += 1
                    sList.append(each_try)
                    print('success count = {}'.format(success_count))
                    break

            print('Episode: {} steps: {}'.format(episode, each_try))
            #if success_count > 100:
            if len(sList) > 10 and np.mean(sList[-10:]) < 200:
                print('Last Episode = []'.format(episode))
                break

            for _ in range(50):
                minibatch = random.sample(replay_buff, 10)
                loss, _ = replay_train(mainDQN, targetDQN, minibatch)
            print('Loss: ', loss)
            sess.run(copy_ops)

        save_path = saver.save(sess, "./Backup/DQN_MountainCar_2015.ckpt")
        print('Model saved in file: %s'%save_path)

        bot_play(mainDQN)

def restore():
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, name='main')
        targetDQN = dqn.DQN(sess, input_size, output_size, name='target')
        saver = tf.train.Saver()
        #saver.restore(sess, "./Backup/DQN_MountainCar_2015_lv2.ckpt")
        saver.restore(sess, "./Backup/DQN_MountainCar_2015_lv4.ckpt")
        #saver.restore(sess, "./Backup/DQN_MountainCar_2015.ckpt")

        bot_play(mainDQN)

def restore_train():
    max_episode = 5000
    max_try = 1000
    replay_buff = deque()

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, name='main')
        targetDQN = dqn.DQN(sess, input_size, output_size, name='target')

        copy_ops = get_copy_var_ops(dest_scope_name='target', src_scope_name='main')

        saver = tf.train.Saver()
        saver.restore(sess, "./Backup/DQN_MountainCar_2015_lv3.ckpt")

        success_count = 0
        sList = []

        for episode in range(max_episode):
            e = 1. / ((episode / 10) +1)
            done = False
            state = env.reset()

            for each_try in range(max_try):
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                next_state, reward, done, _ = env.step(action)
                if done:
                    reward = 1000

                replay_buff.append((state, action, reward, next_state, done))
                if len(replay_buff) > REPLAY_MEMORY:
                    replay_buff.popleft()

                state = next_state

                if done:
                    success_count += 1
                    sList.append(each_try)
                    print('success count = {}'.format(success_count))
                    break

            print('Episode: {} steps: {}'.format(episode, each_try))
            if len(sList) > 10 and np.mean(sList[-10:]) < 120:
                print('Last Episode = []'.format(episode))
                print(sList[-20:])
                break

            for _ in range(50):
                minibatch = random.sample(replay_buff, 10)
                loss, _ = replay_train(mainDQN, targetDQN, minibatch)
            print('Loss: ', loss)
            sess.run(copy_ops)

        save_path = saver.save(sess, "./Backup/DQN_MountainCar_2015_lv4.ckpt")
        print('Model saved in file: %s'%save_path)

        bot_play(mainDQN)



if __name__ == '__main__':
    #main()
    #restore_train()
    restore()




            
        







