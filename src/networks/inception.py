import tensorflow as tf
from tensorflow.contrib import layers

from src.networks.convolutions import conv_1x1, conv_3x3, conv_5x5, deconv_3x3, deconv_5x5

def inception_module(e, out_filters, keep_prob, scope='inception_module'):
  with tf.variable_scope(scope):
    conv_1x1_1 = conv_1x1(
      inputs      = e,
      num_filters = out_filters,
      scope       = 'conv_1x1_1'
    )
    conv_1x1_2 = conv_1x1(
      inputs        = e,
      num_filters   = 8,
      activation_fn = tf.nn.relu,
      scope         = 'conv_1x1_2'
    )
    conv_1x1_3 = conv_1x1(
      inputs        = e,
      num_filters   = 8,
      activation_fn = tf.nn.relu,
      scope         = 'conv_1x1_3'
    )
    conv_3x3_1 = conv_3x3(
      inputs        = conv_1x1_2,
      num_filters   = out_filters,
      activation_fn = None,
      scope         = 'conv_3x3'
    )
    conv_5x5_1 = conv_3x3(
      inputs        = conv_1x1_3,
      num_filters   = out_filters,
      activation_fn = None,
      scope         = 'conv_5x5'
    )
    maxpool = layers.max_pool2d(
      inputs      = e,
      kernel_size = 3,
      stride      = 1,
      padding     = 'SAME',
      scope       = 'max_pool'
    )
    conv_1x1_4 = conv_1x1(
      inputs        = maxpool,
      num_filters   = out_filters,
      activation_fn = None,
      scope         = 'conv_1x1_4'
    )
    
    inception = tf.nn.relu(tf.concat(
      [conv_1x1_1, conv_3x3_1, conv_5x5_1, conv_1x1_4],
      3
    ))

  return inception

def deception_module(e, dim_filters, out_filters, scope='deception_module', final_act = tf.nn.relu):
  with tf.variable_scope(scope):
    conv_1x1_1 = conv_1x1(
      inputs      = e,
      num_filters = out_filters,
      scope       = 'conv_1x1_1'
    )
    conv_1x1_2 = conv_1x1(
      inputs        = e,
      num_filters   = dim_filters,
      activation_fn = tf.nn.relu,
      scope         = 'conv_1x1_2'
    )
    conv_1x1_3 = conv_1x1(
      inputs        = e,
      num_filters   = dim_filters,
      activation_fn = tf.nn.relu,
      scope         = 'conv_1x1_3'
    )
    conv_3x3_1 = deconv_3x3(
      inputs        = conv_1x1_2,
      num_filters   = out_filters,
      activation_fn = None,
      scope         = 'deconv_3x3'
    )
    conv_5x5_1 = deconv_5x5(
      inputs        = conv_1x1_3,
      num_filters   = out_filters,
      activation_fn = None,
      scope         = 'deconv_5x5'
    )
    
    deception = final_act(tf.concat(
      [conv_1x1_1, conv_3x3_1, conv_5x5_1],
      3
    ))


  return deception
