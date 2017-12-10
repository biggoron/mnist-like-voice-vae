import tensorflow as tf

# Gateway
def adversarial_autoencoder(x_hat, x, epsilon, z_sample, vae_elems, dim_img, dim_z, keep_prob, normalize = False):
  encoder = vae_elems['encoder']
  decoder = vae_elems['decoder']
  discriminator = vae_elems['discriminator']

  # encoding
  z_mean, z_stddev = encoder(x_hat, dim_z, keep_prob, normalize = normalize)

  z = tf.add(z_mean, tf.multiply(z_stddev, epsilon), name='z')

  # decoding
  y = decoder(z, dim_img, keep_prob, normalize = normalize)

  # loss
  marginal_likelihood = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(x,y)))
  z_n = tf.norm(z, axis = 1)
  ones = tf.ones_like(z_n)
  dist = tf.squared_difference(z_n, ones)
  marginal_likelihood_2 = dist

  # generator loss
  marginal_likelihood = tf.reduce_mean(marginal_likelihood)
  marginal_likelihood_2 = tf.reduce_mean(marginal_likelihood_2)

  return y, z, marginal_likelihood, marginal_likelihood_2

def decoder(z, decoder, dim_img, normalize):
  y = decoder(z, dim_img, 1.0, reuse=True, normalize = normalize)
  return y
