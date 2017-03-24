# From https://www.tensorflow.org/get_started/get_started
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32, name = 'Weight')
b = tf.Variable([-.3], tf.float32, name = 'Bias')

# Model input and output
x = tf.placeholder(tf.float32, name = 'X-input')
y = tf.placeholder(tf.float32, name = 'Y-input')

linear_model = x * W + b

# cost/loss function
loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares
loss_sum = tf.summary.scalar('loss', loss)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs/LinearRegression', sess.graph)
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
    if i % 20 == 0:
        curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
        print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
        summary = sess.run(merged, feed_dict = {x: x_train, y: y_train})
        writer.add_summary(summary, i)

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))