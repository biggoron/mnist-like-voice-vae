import os
from random import randint

import numpy as np
import tensorflow as tf
from sklearn import manifold
from tqdm import tqdm

from networks.vae_elements import set_vae_elems
import data.mnist_data     as mnist
import utils.prior_factory as prior
import graphs.aae          as aae
import utils.plot_utils    as plot_utils
import data.dataset as dt
import data.dataset2 as dt2

def run_timit(config):

  """ parameters """
  # Setup which encoder and decoder will be used in the program
  # FC only, convolutional, or inception
  global vae_elems
  VAE_ELEMS = set_vae_elems(config['enc_type'], config['dec_type'])

  # Export all the important params to global range
  global ROOT
  ROOT = config['root']

  # Output directories for imgs and models
  global RESULTS_DIR
  RESULTS_DIR = config['results_dir']
  global MODELS_DIR
  MODELS_DIR = config['models_dir']

  # Dimensions of the input/outputs
  global IMG_H # Nb of features
  IMG_H = config['ft_nb']
  global IMG_W # Nb of frames
  IMG_W = config['image_width']
  global DIM_IMG
  DIM_IMG = IMG_H * IMG_W
  global DIM_Z
  DIM_Z = config['dim_z']

  # Train params
  global N_EPOCHS
  N_EPOCHS = config['num_epochs']
  global LEARN_RATE
  LEARN_RATE = config['learn_rate']
  global BATCH_SIZE
  BATCH_SIZE = config['batch_size']

  # Type of target distribution for the embeddings
  global PRIOR_TYPE
  PRIOR_TYPE = config['prior_type']

  # load data
  datastore = dt2.DataStore()
  datastore.dir_to_collection('data/timit_data/', 'all')
  datastore.dir_to_collection('data/dev_mfcc/', 'all')
#  datastore.dir_to_collection('data/test_mfcc/', 'all')
  datastore.shuffle_collection('all')
  dataset = datastore.dataset('all', labels=False, step = 3, ftnb = 29, width = IMG_W)
  n_samples = dataset.elem_nb

  # sampler
  if PRIOR_TYPE == 'hypersphere':
    sampler = prior.hypersphere
  if PRIOR_TYPE == 'normal':
    sampler = prior.normal

  """ build graph """
  # input placeholders
  # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
  x = tf.placeholder(tf.float32, shape=[None, DIM_IMG], name='target_img')
#  x_id = tf.placeholder(tf.float32, shape=[None, 10], name='input_img_label')

  # dropout
  keep_prob = tf.placeholder(tf.float32, name='keep_prob')

  # VAE random element
  epsilon = tf.placeholder(tf.float32, shape=[None, DIM_Z], name='epsilon')

  # input for PMLR
  z_in = tf.placeholder(tf.float32, shape=[None, DIM_Z], name='latent_variable')

  # samples drawn from prior distribution
  z_sample = tf.placeholder(tf.float32, shape=[None, DIM_Z], name='prior_sample')

  # network architecture
  y, z, marginal_likelihood, disc_loss, dist_loss = aae.adversarial_autoencoder(
    x, epsilon, z_sample,
    VAE_ELEMS,
    DIM_IMG, IMG_H, IMG_W, DIM_Z,
    keep_prob
  )

  # optimization
  t_vars = tf.trainable_variables()
#  g_vars = [var for var in t_vars if "encoder" in var.name]
#  dec_vars = [var for var in t_vars if "decoder" in var.name]
#  d_vars = [var for var in t_vars if "discriminator" in var.name]
  ae_vars = [var for var in t_vars if "encoder" or "decoder" in var.name]
#  reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#  regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#  lossL2 = tf.contrib.layers.apply_regularization(regularizer, reg_vars)

  train_ae = tf.train.AdamOptimizer(LEARN_RATE).minimize(
    marginal_likelihood ,
    var_list = ae_vars
  )
#  train_g = tf.train.AdamOptimizer(LEARN_RATE).minimize(
#    marginal_likelihood,
#    var_list = dec_vars
#  )
#  train_distr = tf.train.AdamOptimizer(LEARN_RATE).minimize(
#    dist_loss,
#    var_list = g_vars
#  )
#  train_disc =  tf.train.AdamOptimizer(LEARN_RATE).minimize(
#    disc_loss,
#    var_list = d_vars
#  )
  """ training """

  # train
  total_batch = int(n_samples / BATCH_SIZE)

  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.8})

    saver = tf.train.Saver()

    print_step = 100
    divider1 = print_step
#    divider2 = print_step * DIM_Z
    loss_dist = 1.0
    loss_disc = 1.5

    for epoch in range(N_EPOCHS):

      # Random shuffling
      dataset.shuffle_data()
      dataset.reset_index()

      avg_loss = 0

      # Loop over all batches
      tbar = tqdm(range(total_batch))
      for i in tbar:
        # Compute the offset of the current minibatch in the data.
        offset = (i * BATCH_SIZE) % (n_samples)
        batch_xs_input = dataset.next_batch(BATCH_SIZE, offset, dropout=True, fulls=True)

#        epsilon_sample = prior.hypersphere(BATCH_SIZE * IMG_W, DIM_Z)
        epsilon_sample = np.random.normal(0, 1, [BATCH_SIZE, DIM_Z])

#        if randint(0, int((6. * loss_dist / loss_disc)**2)) == 0:
#          samples = sampler(BATCH_SIZE, DIM_Z)
#          _, loss_dist, loss_disc = sess.run(
#            (train_disc, dist_loss, disc_loss),
#            feed_dict={
#              x: batch_xs_input,
#              z_sample: samples,
#              epsilon: epsilon_sample,
#              keep_prob: 0.8
#            }
#          )
#        if randint(0, int((1. / 2.25) * ((loss_dist / loss_disc) - 5 )**2)) == 0:
#          samples = sampler(BATCH_SIZE, DIM_Z)
#          _, loss_dist, loss_disc = sess.run(
#            (train_distr, dist_loss, disc_loss),
#            feed_dict={
#              x: batch_xs_input,
#              z_sample: samples,
#              epsilon: epsilon_sample,
#              keep_prob: 0.8
#            }
#          )
        samples = sampler(BATCH_SIZE, DIM_Z)
        _, loss_likelihood = sess.run(
          (train_ae, marginal_likelihood),
          feed_dict={
            x: batch_xs_input,
            z_sample: samples,
            epsilon: epsilon_sample,
            keep_prob: 0.8
          }
        )

#        avg_dc_loss += loss_disc
#        avg_dt_loss += loss_distr
        avg_loss += loss_likelihood
        
        if i % print_step == print_step - 1:
          print("epoch %d: L_likelihood %03.6f" % (epoch, avg_loss/divider1))
          avg_loss = 0

      save_path = saver.save(sess, os.path.join(MODELS_DIR, 'save_%i.ckpt' % epoch))
