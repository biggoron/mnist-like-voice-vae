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


"""main function"""
def run_mnist(config):

  """ parameters """
  RESULTS_DIR = config['results_dir']

  # network architecture
  dim_img = config['image_size']**2  # number of pixels for a MNIST image
  dim_z = config['dim_z']                      # to visualize learned manifold

  # train
  n_epochs = config['num_epochs']
  batch_size = config['batch_size']
  learn_rate = config['learn_rate']

  # Plot
  PRR = config['PRR']                              # Plot Reproduce Result
  PRR_n_img_x = config['PRR_n_img_x']              # number of images along x-axis in a canvas
  PRR_n_img_y = config['PRR_n_img_y']              # number of images along y-axis in a canvas
  PRR_resize_factor = config['PRR_resize_factor']  # resize factor for each image in a canvas

  PMLR = config['PMLR']                            # Plot Manifold Learning Result
  PMLR_n_img_x = config['PMLR_n_img_x']            # number of images along x-axis in a canvas
  PMLR_n_img_y = config['PMLR_n_img_y']            # number of images along y-axis in a canvas
  PMLR_resize_factor = config['PMLR_resize_factor']# resize factor for each image in a canvas
  PMLR_z_range = config['PMLR_z_range']            # range for random latent vector
  PMLR_n_samples = config['PMLR_n_samples']        # number of labeled samples to plot a map from input data space to the latent space

  """ prepare MNIST data """
  train_total_data, n_samples, _, _, test_data, test_labels = mnist.prepare_MNIST_data()

  """ build graph """
  # input placeholders
  # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
  x_hat = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
  x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')
#  x_id = tf.placeholder(tf.float32, shape=[None, 10], name='input_img_label')

  # dropout
  keep_prob = tf.placeholder(tf.float32, name='keep_prob')

  # VAE random element
  epsilon = tf.placeholder(tf.float32, shape=[None, dim_z], name='epsilon')

  # input for PMLR
  z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')

  # samples drawn from prior distribution
  z_sample = tf.placeholder(tf.float32, shape=[None, dim_z], name='prior_sample')

  # network architecture
  y, z, marginal_likelihood, marginal_likelihood_2, D_loss, G_loss = aae.adversarial_autoencoder(
    x_hat, x, epsilon, z_sample, config['vae_elems'],
    dim_img, dim_z, keep_prob, config['normalize']
  )

  # optimization
  t_vars = tf.trainable_variables()
  d_vars = [var for var in t_vars if "discriminator" in var.name]
  g_vars = [var for var in t_vars if "encoder" in var.name]
  dec_vars = [var for var in t_vars if "decoder" in var.name]
  ae_vars = [var for var in t_vars if "encoder" or "decoder" in var.name]


  train_op_ae = tf.train.AdamOptimizer(learn_rate).minimize(
    marginal_likelihood,
    var_list = ae_vars
  )
  train_op_distr = tf.train.AdamOptimizer(learn_rate).minimize(
    marginal_likelihood_2,
    var_list = g_vars
  )
  train_op_d = tf.train.AdamOptimizer(learn_rate).minimize(
    D_loss,
    var_list = d_vars
  )
  train_op_g = tf.train.AdamOptimizer(learn_rate).minimize(
    G_loss,
    var_list = g_vars
  )

  train_op_tot = tf.train.AdamOptimizer(learn_rate).minimize(
    marginal_likelihood + marginal_likelihood_2 * 0.001,
    var_list = ae_vars
  )
  """ training """

  # Plot for reproduce performance
  if PRR:
    PRR = plot_utils.Plot_Reproduce_Performance(RESULTS_DIR, PRR_n_img_x, PRR_n_img_y, config['image_size'], config['image_size'], PRR_resize_factor)

    x_PRR = test_data[0:PRR.n_tot_imgs, :]

    x_PRR_img = x_PRR.reshape(PRR.n_tot_imgs, config['image_size'], config['image_size'])
    PRR.save_images(x_PRR_img, name='input.jpg')

  # Plot for manifold learning result
  if PMLR:
    PMLR = plot_utils.Plot_Manifold_Learning_Result(RESULTS_DIR, PMLR_n_img_x, PMLR_n_img_y, config['image_size'], config['image_size'], PMLR_resize_factor, PMLR_z_range)

    x_PMLR = test_data[0:PMLR_n_samples, :]
    id_PMLR = test_labels[0:PMLR_n_samples, :]

    decoded = aae.decoder(z_in, config['vae_elems']['decoder'], dim_img, config['normalize'])

  # train
  total_batch = int(n_samples / batch_size)
  min_tot_loss = 1e99

  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})


    for epoch in range(n_epochs):

      # Random shuffling
      np.random.shuffle(train_total_data)
      train_data_ = train_total_data[:, :-mnist.NUM_LABELS]

      # Loop over all batches
      tbar = tqdm(range(total_batch))
      for i in tbar:
        # Compute the offset of the current minibatch in the data.
        offset = (i * batch_size) % (n_samples)
        batch_xs_input = train_data_[offset:(offset + batch_size), :]
        batch_xs_target = batch_xs_input

        if config['prior_type'] == 'hypersphere':
          samples = prior.hypersphere(batch_size, dim_z)
        if config['prior_type'] == 'normal':
          samples = prior.normal(batch_size, dim_z)

        epsilon_sample = np.random.normal(0, 1, [batch_size, dim_z])


#        while d_loss > g_loss:
#        #for _ in range(4):
#          _, d_loss = sess.run(
#            (train_op_d, D_loss),
#            feed_dict={
#              x_hat: batch_xs_input,
#              x: batch_xs_target,
#              epsilon: epsilon_sample,
#              z_sample: samples,
#              keep_prob: 0.5
#            }
#          )
#          print("epoch %d: L_tot %03.4f L_likelihood %03.4f D_LOSS %03.4f g_loss %03.4f" % (epoch, tot_loss, loss_likelihood, d_loss, g_loss))

        for _ in range(8):
          _, loss_likelihood, loss_distr = sess.run(
            (train_op_tot, marginal_likelihood, marginal_likelihood_2),
            feed_dict={
              x_hat: batch_xs_input,
              x: batch_xs_target,
              epsilon: epsilon_sample,
              z_sample: samples,
              keep_prob: 0.75
            }
          )
        #print("epoch %d: L_tot %03.4f L_likelihood %03.4f d_loss %03.4f g_loss %03.4f" % (epoch, tot_loss, loss_likelihood, d_loss, g_loss))

#        while 0.100 > g_loss:
#        #for _ in range(4):
#          _, loss_likelihood, g_loss, d_loss = sess.run(
#            (train_op_ae, neg_marginal_likelihood, G_loss, D_loss),
#            feed_dict={
#              x_hat: batch_xs_input,
#              x: batch_xs_target,
#              epsilon: epsilon_sample,
#              z_sample: samples,
#              keep_prob: 0.5
#            }
#          )
#          print("epoch %d: L_tot %03.4f L_likelihood %03.4f d_loss %03.4f G_LOSS %03.4f" % (epoch, tot_loss, loss_likelihood, d_loss, g_loss))

        # print cost every epoch
      print("epoch %d: L_likelihood %03.4f distr_loss %03.4f" % (epoch, loss_likelihood, loss_distr))

      # if minimum loss is updated or final epoch, plot results
      if epoch%5==0: # or min_tot_loss > tot_loss or epoch+1 == n_epochs:
        # Plot for reproduce performance
        if PRR:
          epsilon_sample = np.zeros([x_PRR.shape[0], dim_z])
          y_PRR = sess.run(
            y,
            feed_dict={
              x_hat: x_PRR,
              epsilon: epsilon_sample,
              keep_prob : 1
            }
          )
          y_PRR_img = y_PRR.reshape(PRR.n_tot_imgs, config['image_size'], config['image_size'])
          PRR.save_images(y_PRR_img, name="/PRR_epoch_%02d" %(epoch) + ".jpg")

        # Plot for manifold learning result
        if PMLR and dim_z == 2:
          y_PMLR = sess.run(decoded, feed_dict={z_in: PMLR.z, keep_prob : 1})
          y_PMLR_img = y_PMLR.reshape(PMLR.n_tot_imgs, config['image_size'], config['image_size'])
          PMLR.save_images(y_PMLR_img, name="/PMLR_epoch_%02d" % (epoch) + ".jpg")

        if PMLR:
          # plot distribution of labeled images
          epsilon_sample = np.zeros([x_PMLR.shape[0], dim_z])
          z_PMLR = sess.run(
            z,
            feed_dict={
              x_hat: x_PMLR,
              epsilon: epsilon_sample,
              keep_prob : 1
            }
          )
          tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
          try:
            l_tsne = tsne.fit_transform(z_PMLR)
          except ValueError as e:
            print('problem with tsne')
            print(e)
            
          PMLR.save_scattered_image(l_tsne, id_PMLR, name="/PMLR_map_epoch_%02d" % (epoch) + ".jpg")

def run_voice(config):

  """ parameters """
  RESULTS_DIR = config['results_dir']
  MODELS_DIR = config['model_dir']

  # network architecture
  dim_img = config['image_size']**2  # number of pixels for a MNIST image
  dim_z = config['dim_z']                      # to visualize learned manifold

  # train
  n_epochs = config['num_epochs']
  batch_size = config['batch_size']
  learn_rate = config['learn_rate']

  # Plot
  PRR = config['PRR']                              # Plot Reproduce Result
  PRR_n_img_x = config['PRR_n_img_x']              # number of images along x-axis in a canvas
  PRR_n_img_y = config['PRR_n_img_y']              # number of images along y-axis in a canvas
  PRR_resize_factor = config['PRR_resize_factor']  # resize factor for each image in a canvas

  PMLR = config['PMLR']                            # Plot Manifold Learning Result
  PMLR_n_img_x = config['PMLR_n_img_x']            # number of images along x-axis in a canvas
  PMLR_n_img_y = config['PMLR_n_img_y']            # number of images along y-axis in a canvas
  PMLR_resize_factor = config['PMLR_resize_factor']# resize factor for each image in a canvas
  PMLR_z_range = config['PMLR_z_range']            # range for random latent vector
  PMLR_n_samples = config['PMLR_n_samples']        # number of labeled samples to plot a map from input data space to the latent space

  """ prepare MNIST data """
#  train_total_data, n_samples, _, _, test_data, test_labels = mnist.prepare_MNIST_data()


  dataset = dt.VoiceData(config)
  n_samples = dataset.train.elem_nb


  """ build graph """
  # input placeholders
  # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
  x_hat = tf.placeholder(tf.float32, shape=[None, 13 * 13], name='input_img')
  x = tf.placeholder(tf.float32, shape=[None, 13 * 13], name='target_img')
#  x_id = tf.placeholder(tf.float32, shape=[None, 10], name='input_img_label')

  # dropout
  keep_prob = tf.placeholder(tf.float32, name='keep_prob')

  # VAE random element
  epsilon = tf.placeholder(tf.float32, shape=[None, dim_z], name='epsilon')

  # input for PMLR
  z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')

  # samples drawn from prior distribution
  z_sample = tf.placeholder(tf.float32, shape=[None, dim_z], name='prior_sample')

  # network architecture
  y, z, marginal_likelihood, marginal_likelihood_2 = aae.adversarial_autoencoder(
    x_hat, x, epsilon, z_sample, config['vae_elems'],
    dim_img, dim_z, keep_prob, config['normalize']
  )

  # optimization
  t_vars = tf.trainable_variables()
  g_vars = [var for var in t_vars if "encoder" in var.name]
  dec_vars = [var for var in t_vars if "decoder" in var.name]
  ae_vars = [var for var in t_vars if "encoder" or "decoder" in var.name]

  train_op_tot = tf.train.AdamOptimizer(learn_rate).minimize(
    marginal_likelihood,
    var_list = ae_vars
  )
  train_distr = tf.train.AdamOptimizer(learn_rate).minimize(
    marginal_likelihood + marginal_likelihood_2 * 0.1,
    var_list = ae_vars
  )
  """ training """

  # train
  total_batch = int(n_samples / batch_size)

  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.75})

    saver = tf.train.Saver()

    for epoch in range(n_epochs):

      # Random shuffling
      dataset.train.shuffle_data()
      dataset.train.reset_index()

      avg_loss = 0
      avg_d_loss = 0
      loss_distr = 0

      # Loop over all batches
      tbar = tqdm(range(total_batch))
      for i in tbar:
        # Compute the offset of the current minibatch in the data.
        offset = (i * batch_size) % (n_samples)
        batch_xs_input = dataset.train.next_batch(batch_size, offset)
        batch_xs_target = batch_xs_input


        epsilon_sample = np.random.normal(0, 1, [batch_size, dim_z])
        if False: #loss_distr > 5:
          _, loss_likelihood, loss_distr = sess.run(
            (train_distr, marginal_likelihood, marginal_likelihood_2),
            feed_dict={
              x_hat: batch_xs_input,
              x: batch_xs_target,
              epsilon: epsilon_sample,
              keep_prob: 0.75
            }
          )
        if True: # loss_distr < 50:
          _, loss_likelihood, loss_distr = sess.run(
            (train_op_tot, marginal_likelihood, marginal_likelihood_2),
            feed_dict={
              x_hat: batch_xs_input,
              x: batch_xs_target,
              epsilon: epsilon_sample,
              keep_prob: 0.75
            }
          )

        avg_d_loss += loss_distr
        avg_loss += loss_likelihood
        
        if i % 1 == 0:
          print("epoch %d: L_likelihood %03.6f || L_distr %03.6f" % (epoch, avg_loss/16900., avg_d_loss/16900.))
          avg_loss = 0
          avg_d_loss = 0
          

        # print cost every epoch
      print("epoch %d: L_likelihood %03.6f || L_distr %03.6f" % (epoch, avg_loss/16900., avg_d_loss))

      save_path = saver.save(sess, os.path.join(MODELS_DIR, 'save_%i.ckpt' % epoch))


