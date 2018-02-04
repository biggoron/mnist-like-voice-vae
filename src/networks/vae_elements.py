import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers

from src.networks.inception import inception_module, deception_module
from src.networks.convolutions import conv_1x1
from src.networks.discriminator import discriminator

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
    # 1st max pool
    e = layers.max_pool2d(
      inputs      = e,
      kernel_size = 2,
      stride      = 2,
      scope       = 'max_pool_1'
    )

    # 2nd convolution
    e = layers.conv2d(
      inputs            = e,
      num_outputs       = 64,
      kernel_size       = 5,
      stride            = 1,
      activation_fn     = tf.nn.relu,
      scope             = 'conv_2'
    )
    # 2nd max pool
    e = layers.max_pool2d(
      inputs      = e,
      kernel_size = 2,
      stride      = 2,
      scope       = 'max_pool_2'
    )

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
    # 2nd max pool
    e = layers.max_pool2d(
      inputs      = e,
      kernel_size = 3,
      stride      = 1,
      scope       = 'max_pool_3'
    )

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

def inception_encoder(x, h, w, dim_z, keep_prob):
  with tf.variable_scope("inception_encoder"):
    reg = tf.contrib.layers.l2_regularizer(scale=0.0)
    x = tf.reshape(x, [-1, h])
    e = layers.fully_connected(
      inputs              = x,
      num_outputs         = 256,
      activation_fn       = tf.nn.relu,
      weights_regularizer = reg,
      scope               = 'fc_1'
    )
    e = tf.nn.dropout(e, keep_prob)
    e = layers.fully_connected(
      inputs              = e,
      num_outputs         = 128,
      activation_fn       = tf.nn.relu,
      weights_regularizer = reg,
      scope               = 'fc_2'
    )
    e = tf.nn.dropout(e, keep_prob)
    e = layers.fully_connected(
      inputs              = e,
      num_outputs         = 32,
      activation_fn       = tf.nn.sigmoid,
      weights_regularizer = reg,
      scope               = 'fc_3'
    )
    e = tf.nn.dropout(e, keep_prob)
    e = tf.reshape(e, [-1, w, 32, 1])
    i1 = inception_module(e, 16, keep_prob, reg = reg, scope='inception_module_1')
    i2 = inception_module(i1, 16, keep_prob, reg = reg, scope='inception_module_2')
    i3 = inception_module(i2, 16, keep_prob, reg = reg, scope='inception_module_3')
    i3 = layers.flatten(i3)
    
    # 1st hidden layer
    e = layers.fully_connected(
      inputs              = i3,
      num_outputs         = 512,
      activation_fn       = tf.nn.relu,
      weights_regularizer = reg,
      scope               = 'fc_4'
    )
    e = tf.nn.dropout(e, keep_prob)
    e = layers.fully_connected(
      inputs              = i3,
      num_outputs         = 256,
      activation_fn       = tf.nn.relu,
      weights_regularizer = reg,
      scope               = 'fc_5'
    )
    e = tf.nn.dropout(e, keep_prob)

    # output layer
    e1 = layers.fully_connected(
      inputs              = e,
      num_outputs         = dim_z,
      activation_fn       = None,
      scope               = 'fc_stddev_output_layer'
    )
    
    e2 = layers.fully_connected(
      inputs              = e,
      num_outputs         = dim_z,
      activation_fn       = None,
      scope               = 'fc_mean_output_layer'
    )

    return e2, e1

def deception_decoder(z, h, w, dim_img, keep_prob, reuse=False):
  with tf.variable_scope("conv_decoder", reuse=reuse):
    reg = tf.contrib.layers.l2_regularizer(scale=0.0)
    e = layers.fully_connected(
      inputs              = z,
      num_outputs         = 128,
      activation_fn       = tf.nn.relu,
      weights_regularizer = reg,
      scope               = 'first_fc_layer'
    )
    e = tf.nn.dropout(e, keep_prob)
    e = layers.fully_connected(
      inputs              = z,
      num_outputs         = 256,
      activation_fn       = tf.nn.relu,
      weights_regularizer = reg,
      scope               = 'second_fc_layer'
    )
    e = tf.nn.dropout(e, keep_prob)
    e = layers.fully_connected(
      inputs              = z,
      num_outputs         = dim_img*32,
      activation_fn       = tf.nn.relu,
      weights_regularizer = reg,
      scope               = 'third_fc_layer'
    )
    e = tf.nn.dropout(e, keep_prob)

    e = tf.reshape(e, [-1, w, h, 32])

    i2 = deception_module(e, 8, 32, 'deception_2', reg = reg)
    i3 = deception_module(i2, 8, 16, 'deception_3', reg = reg)
    i4 = deception_module(i3, 8, 8, 'deception_4', reg = reg)
    i5 = deception_module(i4, 1, 1, 'deception_5', reg = reg)

    e = layers.conv2d_transpose(
      inputs            = i5,
      num_outputs       = 1,
      kernel_size       = 1,
      activation_fn     = tf.nn.tanh,
      scope             = 'deconv_5'
    )

    e = tf.reshape(e, [-1, dim_img])

  return e

def rnn_encoder(x, h, w, dim_z, keep_prob):
  with tf.variable_scope("rnn_encoder"):
    x = tf.reshape(x, [-1, h])
    e = layers.fully_connected(
      inputs              = x,
      num_outputs         = 256,
      activation_fn       = tf.nn.relu,
      scope               = 'fc_1'
    )
    e = tf.nn.dropout(e, keep_prob)
    e = layers.fully_connected(
      inputs              = e,
      num_outputs         = 64,
      activation_fn       = tf.nn.relu,
      scope               = 'fc_2'
    )
    e = tf.nn.dropout(e, keep_prob)
    e = layers.fully_connected(
      inputs              = e,
      num_outputs         = 32,
      activation_fn       = tf.nn.sigmoid,
      scope               = 'fc_3'
    )
    ee = e
    e = tf.nn.dropout(e, keep_prob)
    e = tf.reshape(e, [-1, w, 32])
    e = tf.unstack(e, num = w, axis = 1)
    cells = [rnn.BasicLSTMCell(32, forget_bias=1.0) for _ in range(3)]
    stack = rnn.MultiRNNCell(cells, state_is_tuple=True)
    e, final_state = rnn.static_rnn(stack, e, dtype=tf.float32)
    e = tf.stack(e, axis=1)
    e = tf.reshape(e, [-1, 32])
    e = tf.nn.dropout(e, keep_prob)
    e1 = layers.fully_connected(
      inputs              = e,
      num_outputs         = dim_z,
      activation_fn       = tf.nn.sigmoid,
      scope               = 'fc_6'
    )
    e2 = layers.fully_connected(
      inputs              = e,
      num_outputs         = dim_z,
      activation_fn       = tf.nn.sigmoid,
      scope               = 'fc_7'
    )
    e1 = tf.reshape(e1, [-1, w, dim_z])
    e2 = tf.reshape(e2, [-1, w, dim_z])
    return e1, e2

def rnn_decoder(z, h, w, dim_z, keep_prob, reuse = False):
  with tf.variable_scope("rnn_decoder"):
    z = tf.reshape(z, [-1, dim_z])
    e = layers.fully_connected(
      inputs              = z,
      num_outputs         = 256,
      activation_fn       = tf.nn.relu,
      scope               = 'fc_1'
    )
    e = tf.nn.dropout(e, keep_prob)
    e = layers.fully_connected(
      inputs              = e,
      num_outputs         = 128,
      activation_fn       = tf.nn.relu,
      scope               = 'fc_2'
    )
    e = tf.nn.dropout(e, keep_prob)
    e = layers.fully_connected(
      inputs              = e,
      num_outputs         = h*8,
      activation_fn       = tf.nn.sigmoid,
      scope               = 'fc_3'
    )
    e = tf.reshape(e, [-1, w, h, 8])

    # 1st deconv layer
    e = layers.conv2d_transpose(
      inputs            = e,
      num_outputs       = 32,
      kernel_size       = 5,
      padding           = 'SAME',
      activation_fn     = tf.nn.relu,
      scope             = 'deconv_1'
    )
    print(e.get_shape().as_list())

    # 2nd deconv layer
    e = layers.conv2d_transpose(
      inputs            = e,
      num_outputs       = 8,
      kernel_size       = 1,
      stride            = 1,
      padding           = 'SAME',
      activation_fn     = tf.nn.relu,
      scope             = 'deconv_2'
    )
    print(e.get_shape().as_list())

    # 3rd deconv layer
    e = layers.conv2d_transpose(
      inputs            = e,
      num_outputs       = 8,
      kernel_size       = 3,
      stride            = 1,
      padding           = 'SAME',
      activation_fn     = tf.nn.relu,
      scope             = 'deconv_3'
    )
    print(e.get_shape().as_list())

    # 4th deconv layer
    e = layers.conv2d_transpose(
      inputs            = e,
      num_outputs       = 3,
      kernel_size       = 3,
      stride            = 1,
      padding           = 'SAME',
      activation_fn     = tf.nn.relu,
      scope             = 'deconv_4'
    )
    print(e.get_shape().as_list())
#    i3 = deception_module(e, 2, 2, 'deception_3')
#    i4 = deception_module(i3, 2, 2, 'deception_4')
#    i5 = deception_module(i4, 1, 1, 'deception_5')

    e = layers.conv2d_transpose(
      inputs            = e,
      num_outputs       = 1,
      kernel_size       = 1,
      padding           = 'SAME',
      activation_fn     = None,
      scope             = 'deconv_5'
    )

    e = tf.reshape(e, [-1, h * w])

    return e
  
def set_vae_elems(enc, dec):
  vae_elems = {}
  encs = ['fc','conv','inception','rnn']
  decs = ['fc','conv','inception','rnn']
  try:
    assert enc in encs
    if enc == 'fc':
      vae_elems['encoder'] = fc_encoder
    elif enc == 'conv':
      vae_elems['encoder'] = conv_encoder
    elif enc == 'inception':
      vae_elems['encoder'] = inception_encoder
    elif enc == 'rnn':
      vae_elems['encoder'] = rnn_encoder
  except :
    print('vae must use either fully connected layers (enc_type = fc) or convolutional architecture (enc_type = conv, enc_type = inception)')

  try:
    assert dec in decs
    if dec == 'fc':
      vae_elems['decoder'] = fc_decoder
    elif dec == 'conv':
      vae_elems['decoder'] = conv_decoder
    elif dec == 'inception':
      vae_elems['decoder'] = deception_decoder
    elif dec == 'rnn':
      vae_elems['decoder'] = rnn_decoder
  except :
    print('vae must use either fully connected layers (dec_type = fc) or convolutional architecture (dec_type = conv)')

  vae_elems['discriminator'] = discriminator

  return vae_elems

