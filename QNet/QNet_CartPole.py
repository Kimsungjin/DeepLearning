import gym
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
lr = 0.1

X = tf.placeholder(shape = [None, input_size], dtype = tf.float32)
W = tf.get_variable("W1", shape = [input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())

Qpred = tf.matmul(X, W)
Y = tf.placeholder(shape= [None, output_size], dtype=tf.float32)

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)

dis = .9
num_episode = 2000
rList = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_episode):
        e = 1./((i/50) + 10)
        rAll = 0
        step_count = 0
        s = env.reset()
        done = False

        while not done:
            step_count += 1
            x = np.reshape(s, [1, input_size])
            Qs = sess.run(Qpred, feed_dict={X: x})
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)

            s1, reward, done, _ = env.step(a)
            if done:
                Qs[0, a] = -100
            else:
                x1 = np.reshape(s1, [1, input_size])
                Qs1 = sess.run(Qpred, feed_dict={X: x1})
                Qs[0, a] = reward + dis * np.max(Qs1)

            sess.run(train, feed_dict = {X: x, Y:Qs})
            s = s1

        rList.append(step_count)
        print('Episode: {} steps: {}'.format(i, step_count))
        if len(rList) > 10 and np.mean(rList[-10:]) > 500:
            break

    observation = env.reset()
    reward_sum = 0
    while True:
        env.render()
        x = np.reshape(observation, [1, input_size])
        Qs = sess.run(Qpred, feed_dict={X: x})
        a = np.argmax(Qs)
        observation, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print('Total score: {}'.format(reward_sum))
            break





