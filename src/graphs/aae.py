import tensorflow as tf

# Gateway
def adversarial_autoencoder(
   x, epsilon, z_sample,
   vae_elems,
   dim_img, h, w, dim_z,
   keep_prob
):

  encoder = vae_elems['encoder']
  decoder = vae_elems['decoder']
  discriminator = vae_elems['discriminator']

  # encoding
  x_cut = tf.reshape(x, [-1, w, h])
  x_cut = x_cut[:, 8:18, :]
  x_cut = tf.reshape(x_cut, [-1, 10 * h])
  z_mean, z_stddev = encoder(x_cut, h, 10, dim_z, keep_prob)

  #  epsilon_r = tf.reshape(epsilon, [-1, w, dim_z])
  
  z = tf.add(z_mean, tf.multiply(z_stddev, epsilon), name='z')

#  vae_logits = discriminator(z, keep_prob)
#  disc_logits = discriminator(z_sample, keep_prob, reuse = True)

#  obj_disc_from_vae = tf.reduce_mean(
#    tf.nn.sigmoid_cross_entropy_with_logits(
#      logits = vae_logits,
#      labels = tf.zeros_like(vae_logits)
#    )
#  )
#  obj_gen_from_vae = tf.reduce_mean(
#    tf.nn.sigmoid_cross_entropy_with_logits(
#      logits = vae_logits,
#      labels = tf.ones_like(vae_logits)
#    )
#  )
#  obj_disc_from_inputs =  tf.reduce_mean(
#    tf.nn.sigmoid_cross_entropy_with_logits(
#      logits = disc_logits,
#      labels = tf.ones_like(disc_logits)
#    )
#  )
#  obj_disc = obj_disc_from_vae + obj_disc_from_inputs

  # decoding
  y = decoder(z, h, w, dim_img, keep_prob)
#  y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

  # loss
  marginal_likelihood = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(x, y)))

  # generator loss
  marginal_likelihood = tf.reduce_mean(marginal_likelihood)

  return y, z, marginal_likelihood, 0, 0#obj_disc, obj_gen_from_vae

def decoder(z, decoder, dim_img, normalize):
  y = decoder(z, dim_img, 1.0, reuse=True, normalize = normalize)
  return y
