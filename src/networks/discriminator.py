import tensorflow as tf

# Discriminator
def discriminator(z, n_output, keep_prob, reuse=False):
  with tf.variable_scope("discriminator", reuse=reuse):
    # initializers
    w_init = tf.contrib.layers.xavier_initializer()
    b_init = tf.constant_initializer(0.)

    # TODO make a better structure (256, 512, 256, output) par exemple
    # 1st hidden layer
    w0 = tf.get_variable('w0', [z.get_shape()[1], 1024], initializer=w_init)
    b0 = tf.get_variable('b0', [1024], initializer=b_init)
    h0 = tf.matmul(z, w0) + b0
    h0 = tf.nn.relu(h0)
    h0 = tf.nn.dropout(h0, keep_prob)

    # 2nd hidden layer
    w1 = tf.get_variable('w1', [h0.get_shape()[1], 1024], initializer=w_init)
    b1 = tf.get_variable('b1', [1024], initializer=b_init)
    h1 = tf.matmul(h0, w1) + b1
    h1 = tf.nn.relu(h1)
    h1 = tf.nn.dropout(h1, keep_prob)

    # output layer
    wo = tf.get_variable('wo', [h1.get_shape()[1], 1], initializer=w_init)
    bo = tf.get_variable('bo', [1], initializer=b_init)
    y = tf.matmul(h1, wo) + bo

  return tf.sigmoid(y), y