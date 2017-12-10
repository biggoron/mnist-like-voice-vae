import tensorflow as tf
from tensorflow.contrib import layers

    
def conv_1x1(inputs, num_filters, activation_fn = None, scope = 'conv_1x1'):
  return layers.conv2d(
    inputs = inputs,
    num_outputs = num_filters,
    kernel_size = 1,
    stride = 1,
    activation_fn = activation_fn,
    scope = scope
  )

def conv_3x3(inputs, num_filters, activation_fn = None, scope = 'conv_3x3'):
  return layers.conv2d(
    inputs = inputs,
    num_outputs = num_filters,
    kernel_size = 3,
    stride = 1,
    activation_fn = activation_fn,
    scope = scope
  )

def conv_5x5(inputs, num_filters, activation_fn = None, scope = 'conv_3x3'):
  return layers.conv2d(
    inputs = inputs,
    num_outputs = num_filters,
    kernel_size = 5,
    stride = 1,
    activation_fn = activation_fn,
    scope = scope
  )

def deconv_3x3(inputs, num_filters, activation_fn = None, scope = 'deconv_3x3'):
  return layers.conv2d_transpose(
    inputs = inputs,
    num_outputs = num_filters,
    kernel_size = 3,
    padding = 'SAME',
    activation_fn = activation_fn,
    scope = scope
  )

def deconv_5x5(inputs, num_filters, activation_fn = None, scope = 'deconv_5x5'):
  return layers.conv2d_transpose(
    inputs = inputs,
    num_outputs = num_filters,
    kernel_size = 5,
    padding = 'SAME',
    activation_fn = activation_fn,
    scope = scope
  )
