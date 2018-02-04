import tensorflow as tf

# Discriminator
def discriminator(z, keep_prob, reuse=False):
  with tf.variable_scope("discriminator", reuse=reuse):
    # initializers
    w_init = tf.contrib.layers.xavier_initializer()
    b_init = tf.constant_initializer(0.)

    # TODO make a better structure (256, 512, 256, output) par exemple
    # 1st hidden layer
    w0 = tf.get_variable('w0', [z.get_shape()[1], 256], initializer=w_init)
    b0 = tf.get_variable('b0', [256], initializer=b_init)
    h0 = tf.matmul(z, w0) + b0
    h0 = tf.nn.relu(h0)
    h0 = tf.nn.dropout(h0, keep_prob)

    # 2nd hidden layer
    w1 = tf.get_variable('w1', [h0.get_shape()[1], 512], initializer=w_init)
    b1 = tf.get_variable('b1', [512], initializer=b_init)
    h1 = tf.matmul(h0, w1) + b1
    h1 = tf.nn.relu(h1)
    h1 = tf.nn.dropout(h1, keep_prob)

    # 3nd hidden layer
    w2 = tf.get_variable('w2', [h1.get_shape()[1], 256], initializer=w_init)
    b2 = tf.get_variable('b2', [256], initializer=b_init)
    h2 = tf.matmul(h1, w2) + b2
    h2 = tf.nn.relu(h2)
    h2 = tf.nn.dropout(h2, keep_prob)

    # output layer
    wo = tf.get_variable('wo', [h2.get_shape()[1], 1], initializer=w_init)
    bo = tf.get_variable('bo', [1], initializer=b_init)
    y = tf.matmul(h2, wo) + bo

  return y
