import tensorflow as tf
from tensorflow.contrib import layers

from src.networks.inception import inception_module, deception_module
from src.networks.convolutions import conv_1x1

def fc_encoder(x, n_output, keep_prob, normalize = False):
  with tf.variable_scope("fc_encoder"):

    if normalize:
      n_fn = layers.batch_norm
      n_pa = {'scale': True, 'scope': 'batch_norm'}
    else:
      n_fn = None
      n_pa = None

    # 1st hidden layer
    h = layers.fully_connected(
      inputs              = x,
      num_outputs         = 1024,
      activation_fn       = tf.nn.relu,
      normalizer_fn       = n_fn,
      normalizer_params   = n_pa,
      scope               = 'first_layer'
    )
    h = tf.nn.dropout(h, keep_prob)

    # 2nd hidden layer
    h = layers.fully_connected(
      inputs              = h,
      num_outputs         = 512,
      activation_fn       = tf.nn.relu,
      normalizer_fn       = n_fn,
      normalizer_params   = n_pa,
      scope               = 'second_layer'
    )
    h = tf.nn.dropout(h, keep_prob)

    # 3rd hidden layer
    h = layers.fully_connected(
      inputs              = h,
      num_outputs         = 256,
      activation_fn       = tf.nn.relu,
      normalizer_fn       = n_fn,
      normalizer_params   = n_pa,
      scope               = 'third_layer'
    )
    h = tf.nn.dropout(h, keep_prob)

    # stddev hidden layer
    h_stddev = layers.fully_connected(
      inputs              = h,
      num_outputs         = 256,
      activation_fn       = tf.nn.relu,
      normalizer_fn       = n_fn,
      normalizer_params   = n_pa,
      scope               = 'stddev_layer'
    )
    h_stddev = tf.nn.dropout(h_stddev, keep_prob)

    # mean hidden layer
    h_mean = layers.fully_connected(
      inputs              = h,
      num_outputs         = 256,
      activation_fn       = tf.nn.relu,
      normalizer_fn       = n_fn,
      normalizer_params   = n_pa,
      scope               = 'mean_layer'
    )
    h_mean = tf.nn.dropout(h_mean, keep_prob)

    # var output layer
    stddev = layers.fully_connected(
      inputs              = h_stddev,
      num_outputs         = n_output,
      activation_fn       = None,
      scope               = 'stddev_stddev_layer'
    )

    # mean output layer
    mean = layers.fully_connected(
      inputs              = h_mean,
      num_outputs         = n_output,
      activation_fn       = None,
      scope               = 'mean_stddev_layer'
    )

  return mean, stddev

def fc_decoder(z, dim_img, keep_prob, reuse=False, normalize = False):
  with tf.variable_scope("fc_decoder", reuse=reuse):
    if normalize:
      n_fn = layers.batch_norm
      n_pa = {'scale': True, 'scope': 'batch_norm'}
    else:
      n_fn = None
      n_pa = None

    # 1st hidden layer
    h = layers.fully_connected(
      inputs              = z,
      num_outputs         = 128,
      activation_fn       = tf.nn.relu,
      normalizer_fn       = n_fn,
      normalizer_params   = n_pa,
      scope               = 'first_layer'
    )
    h = tf.nn.dropout(h, keep_prob)

    # 2nd hidden layer
    h = layers.fully_connected(
      inputs              = h,
      num_outputs         = 1024,
      activation_fn       = tf.nn.relu,
      normalizer_fn       = n_fn,
      normalizer_params   = n_pa,
      scope               = 'second_layer'
    )
    h = tf.nn.dropout(h, keep_prob)

    # 3rd hidden layer
    h = layers.fully_connected(
      inputs              = h,
      num_outputs         = 1024,
      activation_fn       = tf.nn.relu,
      normalizer_fn       = n_fn,
      normalizer_params   = n_pa,
      scope               = 'third_layer'
    )
    h = tf.nn.dropout(h, keep_prob)

    # output layer
    y = layers.fully_connected(
      inputs              = h,
      num_outputs         = dim_img,
      activation_fn       = None,
      scope               = 'output_layer'
    )
  return y

def conv_encoder(x, dim_z, keep_prob, normalize = False):
  with tf.variable_scope("conv_encoder"):
    e = tf.reshape(x, [-1, 28, 28, 1])

    # 1st convolution
    e = layers.conv2d(
      inputs            = e,
      num_outputs       = 32,
      kernel_size       = 5,
      stride            = 1,
      activation_fn     = tf.nn.relu,
      scope             = 'conv_1'
    )
    print(e.get_shape().as_list())
    # 1st max pool
    e = layers.max_pool2d(
      inputs      = e,
      kernel_size = 2,
      stride      = 2,
      scope       = 'max_pool_1'
    )
    print(e.get_shape().as_list())

    # 2nd convolution
    e = layers.conv2d(
      inputs            = e,
      num_outputs       = 64,
      kernel_size       = 5,
      stride            = 1,
      activation_fn     = tf.nn.relu,
      scope             = 'conv_2'
    )
    print(e.get_shape().as_list())
    # 2nd max pool
    e = layers.max_pool2d(
      inputs      = e,
      kernel_size = 2,
      stride      = 2,
      scope       = 'max_pool_2'
    )
    print(e.get_shape().as_list())

    # 3rd convolution
    e = layers.conv2d(
      inputs            = e,
      num_outputs       = 128,
      kernel_size       = 3,
      stride            = 2,
      padding           = 'VALID',
      activation_fn     = tf.nn.relu,
      scope             = 'conv_3'
    )
    print(e.get_shape().as_list())
    # 2nd max pool
    e = layers.max_pool2d(
      inputs      = e,
      kernel_size = 3,
      stride      = 1,
      scope       = 'max_pool_3'
    )
    print(e.get_shape().as_list())

    e = layers.flatten(e)
    e = tf.nn.dropout(e, keep_prob)

    # 1st fc layer
    e = layers.fully_connected(
      inputs              = e,
      num_outputs         = 1024,
      activation_fn       = tf.nn.relu,
      scope               = 'fc_1'
    )
    e = tf.nn.dropout(e, keep_prob)

    # 2nd fc layer
    e = layers.fully_connected(
      inputs              = e,
      num_outputs         = 512,
      activation_fn       = tf.nn.relu,
      scope               = 'fc_2'
    )
    e = tf.nn.dropout(e, keep_prob)

    # 2nd fc layer
    e = layers.fully_connected(
      inputs              = e,
      num_outputs         = 256,
      activation_fn       = tf.nn.relu,
      scope               = 'fc_3'
    )
    e = tf.nn.dropout(e, keep_prob)

    # avg fc layer
    e1 = layers.fully_connected(
      inputs              = e,
      num_outputs         = 256,
      activation_fn       = tf.nn.relu,
      scope               = 'fc_avg'
    )
    e1 = tf.nn.dropout(e1, keep_prob)

    # avg output layer
    e1 = layers.fully_connected(
      inputs              = e1,
      num_outputs         = dim_z,
      scope               = 'avg_output'
    )

    # dev fc layer
    e2 = layers.fully_connected(
      inputs              = e,
      num_outputs         = 256,
      activation_fn       = tf.nn.relu,
      scope               = 'fc_stddev'
    )
    e2 = tf.nn.dropout(e2, keep_prob)

    # stddev output layer
    e2 = layers.fully_connected(
      inputs              = e2,
      num_outputs         = dim_z,
      scope               = 'stddev_output'
    )

  return e1, e2

def conv_decoder(z, dim_img, keep_prob, normalize = False, reuse=False):
  with tf.variable_scope("conv_decoder", reuse=reuse):
    e = layers.fully_connected(
      inputs              = z,
      num_outputs         = 512,
      activation_fn       = tf.nn.relu,
      scope               = 'first_fc_layer'
    )
    e = tf.nn.dropout(e, keep_prob)
    e = tf.expand_dims(tf.expand_dims(e, 1), 1)

    # 1st deconv layer
    e = layers.conv2d_transpose(
      inputs            = e,
      num_outputs       = 128,
      kernel_size       = 3,
      padding           = 'VALID',
      activation_fn     = tf.nn.relu,
      scope             = 'deconv_1'
    )

    # 2nd deconv layer
    e = layers.conv2d_transpose(
      inputs            = e,
      num_outputs       = 64,
      kernel_size       = 3,
      stride            = 2,
      padding           = 'VALID',
      activation_fn     = tf.nn.relu,
      scope             = 'deconv_2'
    )

    # 3rd deconv layer
    e = layers.conv2d_transpose(
      inputs            = e,
      num_outputs       = 32,
      kernel_size       = 3,
      stride            = 2,
      activation_fn     = tf.nn.relu,
      scope             = 'deconv_3'
    )

    # 4th deconv layer
    e = layers.conv2d_transpose(
      inputs            = e,
      num_outputs       = 8,
      kernel_size       = 3,
      stride            = 2,
      activation_fn     = tf.nn.relu,
      scope             = 'deconv_4'
    )

    # 5th deconv layer
    e = layers.conv2d_transpose(
      inputs            = e,
      num_outputs       = 1,
      kernel_size       = 5,
      activation_fn     = tf.nn.tanh,
      scope             = 'deconv_5'
    )
    
    e = tf.reshape(e, [-1, 13, 13])
  return e 

def inception_encoder(x, dim_z, keep_prob, normalize = False):
  with tf.variable_scope("inception_encoder"):
    e = tf.reshape(x, [-1, 13, 13, 1])
#    e = layers.batch_norm(e, scale = True, scope = 'batch_norm')
    i1 = inception_module(e, 16, keep_prob, scope='inception_module_1')
    i2 = inception_module(i1, 16, keep_prob, scope='inception_module_2')
    i3 = inception_module(i2, 16, keep_prob, scope='inception_module_3')
    i4 = inception_module(i3, 16, keep_prob, scope='inception_module_4')
    i5 = inception_module(i4, 16, keep_prob, scope='inception_module_5')
    
    i5 = layers.flatten(i5)
    
    # 1st hidden layer
    e = layers.fully_connected(
      inputs              = i5,
      num_outputs         = 512,
      activation_fn       = tf.nn.relu,
      scope               = 'fc_layer'
    )
    e = tf.nn.dropout(e, keep_prob)

    # output layer
    e1 = layers.fully_connected(
      inputs              = e,
      num_outputs         = dim_z,
      activation_fn       = None,
      scope               = 'stddev_output_layer'
    )
    
    e2 = layers.fully_connected(
      inputs              = e,
      num_outputs         = dim_z,
      activation_fn       = None,
      scope               = 'mean_output_layer'
    )

    return e2, e1

def deception_decoder(z, dim_img, keep_prob, normalize = False, reuse=False):
  with tf.variable_scope("conv_decoder", reuse=reuse):

    e = layers.fully_connected(
      inputs              = z,
      num_outputs         = 32,
      activation_fn       = tf.nn.relu,
      scope               = 'first_fc_layer'
    )
    e = tf.nn.dropout(e, keep_prob)
    e = layers.fully_connected(
      inputs              = z,
      num_outputs         = 64,
      activation_fn       = tf.nn.relu,
      scope               = 'second_fc_layer'
    )
    e = tf.nn.dropout(e, keep_prob)
    e = layers.fully_connected(
      inputs              = z,
      num_outputs         = 13*13*32,
      activation_fn       = tf.nn.relu,
      scope               = 'third_fc_layer'
    )
    e = tf.nn.dropout(e, keep_prob)

    e = tf.reshape(e, [-1, 13, 13, 32])

    i1 = deception_module(e, 16, 64, 'deception_1')
    i2 = deception_module(i1, 8, 32, 'deception_2')
    i3 = deception_module(i2, 8, 16, 'deception_3')
    i4 = deception_module(i3, 8, 8, 'deception_4')
    i5 = deception_module(i4, 1, 1, 'deception_5')

    e = layers.conv2d_transpose(
      inputs            = i5,
      num_outputs       = 1,
      kernel_size       = 5,
      activation_fn     = tf.nn.tanh,
      scope             = 'deconv_5'
    )

    e = tf.reshape(e, [-1, 13*13])

  return e
