import os
import tensorflow as tf
import numpy as np

class Graph():
  def __init__(self, graph_dir, graph_file):
    self.sess=tf.Session()    
    #First let's load meta graph and restore weights
    self.saver = tf.train.import_meta_graph(os.path.join(graph_dir, graph_file))
    self.saver.restore(self.sess,tf.train.latest_checkpoint(graph_dir))

    self.graph = tf.get_default_graph()

    self.x_target = self.graph.get_tensor_by_name("target_img:0")
    self.epsilon = self.graph.get_tensor_by_name("epsilon:0")
    self.keep_prob = self.graph.get_tensor_by_name("keep_prob:0")
    self.z = self.graph.get_tensor_by_name("z:0")

  def embed(self, dataset, batch_size, points = [], labels = [], with_labels = False):
    n_samples = dataset.elem_nb
    total_batch = int(n_samples / batch_size)
    points = []

    for i in range(total_batch):
      # Compute the offset of the current minibatch in the data.
      offset = (i * batch_size) % (n_samples)
      if with_labels:
        batch_xs_input, batch_labels = dataset.next_batch(batch_size, offset)
      else:
        batch_xs_input = dataset.next_batch(batch_size, offset)

      epsilon_sample = np.zeros([batch_xs_input.shape[0], 20])
      z_res = self.sess.run(
        self.z, 
        feed_dict={
          self.x_target: batch_xs_input,
          self.epsilon: epsilon_sample,
          self.keep_prob : 1.0
        }
      )
      for s in z_res:
        points.append(s)
      if with_labels:
        for l in batch_labels:
          labels.append(l)
    points = np.asarray(points)
    if with_labels:
      labels = np.asarray(labels)
      return points, labels
    else:
      return points
