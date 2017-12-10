import os
from matplotlib.collections import LineCollection

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from sklearn import manifold

from src.utils.load_conf import parse_args, parse_conf
import src.data.dataset as dt

def test_1(config):
  dataset = dt.VoiceData(config)
  n_samples = dataset.test.elem_nb
  batch_size = config['batch_size']

  sess=tf.Session()    
  #First let's load meta graph and restore weights
  saver = tf.train.import_meta_graph('saved_models/complicated_no_distr/save_3.ckpt.meta')
  saver.restore(sess,tf.train.latest_checkpoint('saved_models/complicated_no_distr/'))

  graph = tf.get_default_graph()
#  print('here')
#  print('\n'.join([n.name for n in tf.get_default_graph().as_graph_def().node if 'Add' in n.name]))
  train_writer = tf.summary.FileWriter('src/tests/test_1/', graph)
  x_hat = graph.get_tensor_by_name("input_img:0")
  x_target = graph.get_tensor_by_name("target_img:0")
  epsilon = graph.get_tensor_by_name("epsilon:0")
  keep_prob = graph.get_tensor_by_name("keep_prob:0")
  z = graph.get_tensor_by_name("z:0")


  total_batch = int(n_samples / conf['batch_size'])
  tbar = tqdm(range(total_batch))

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  points = []
  for i in tbar:
    # Compute the offset of the current minibatch in the data.
    offset = (i * batch_size) % (n_samples)
    batch_xs_input = dataset.test.next_batch(batch_size, offset)
    batch_xs_target = batch_xs_input

    epsilon_sample = np.zeros([batch_xs_input.shape[0], 40])
    z_res = sess.run(
      z, 
      feed_dict={
        x_hat: batch_xs_input,
#        x_target: batch_xs_input,
        epsilon: epsilon_sample,
        keep_prob : 1.0
      }
    )
    for s in z_res:
      points.append(s)
  points = np.asarray(points)
  print(points[:, 0])
  tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
  l_tsne = tsne.fit_transform(points)
  ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')

  plt.show()

def test_2(config, test_dir, fn):
  dataset = dt.VoiceData(config)
  n_samples = dataset.test.elem_nb
  batch_size = config['batch_size']

  sess=tf.Session()    
  #First let's load meta graph and restore weights
  saver = tf.train.import_meta_graph('saved_models/complicated_no_distr/save_3.ckpt.meta')
  saver.restore(sess,tf.train.latest_checkpoint('saved_models/complicated_no_distr/'))

  graph = tf.get_default_graph()
#  print('here')
#  print('\n'.join([n.name for n in tf.get_default_graph().as_graph_def().node if 'Add' in n.name]))
  train_writer = tf.summary.FileWriter('src/tests/test_1/', graph)
  x_hat = graph.get_tensor_by_name("input_img:0")
  x_target = graph.get_tensor_by_name("target_img:0")
  epsilon = graph.get_tensor_by_name("epsilon:0")
  keep_prob = graph.get_tensor_by_name("keep_prob:0")
  z = graph.get_tensor_by_name("z:0")


  total_batch = int(n_samples / conf['batch_size'])
  tbar = tqdm(range(total_batch))

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  points = []
  for i in tbar:
    # Compute the offset of the current minibatch in the data.
    offset = (i * batch_size) % (n_samples)
    batch_xs_input = dataset.test.next_batch(batch_size, offset)
    batch_xs_target = batch_xs_input

    epsilon_sample = np.zeros([batch_xs_input.shape[0], 40])
    z_res = sess.run(
      z, 
      feed_dict={
        x_hat: batch_xs_input,
#        x_target: batch_xs_input,
        epsilon: epsilon_sample,
        keep_prob : 1.0
      }
    )
    for s in z_res:
      points.append(s)
  points = np.asarray(points)
  print(points[:, 0])
  tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
  l_tsne = tsne.fit_transform(points)
#  ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='whitesmoke', marker='o')

  dataset = dt.FileData(test_dir, fn)
  n_samples = dataset.elem_nb

  total_batch = int(n_samples / conf['batch_size'])
  tbar = tqdm(range(total_batch))

  points = []
  for i in tbar:
    # Compute the offset of the current minibatch in the data.
    offset = (i * batch_size) % (n_samples)
    batch_xs_input = dataset.next_batch(batch_size, offset)
    batch_xs_target = batch_xs_input

    epsilon_sample = np.zeros([batch_xs_input.shape[0], 40])
    z_res = sess.run(
      z, 
      feed_dict={
        x_hat: batch_xs_input,
        epsilon: epsilon_sample,
        keep_prob : 1.0
      }
    )
    for s in z_res:
      points.append(s)
  points = np.asarray(points)
  print(points[:, 0])
  l_tsne = tsne.fit_transform(points)
  x = l_tsne[:,0]
  y = l_tsne[:,1]
  points = np.array([x, y]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  l = len(segments)
  t = np.linspace(0, 1, l)
  #plt.plot(x, y, '-o', c=col, cmap=plt.get_cmap("plasma"), markersize=2)
  lc = LineCollection(segments, cmap=plt.get_cmap('plasma'))
  lc.set_array(t)
  lc.set_linewidth(2)
  fig = plt.figure(figsize = (20, 20), dpi = 500)
  plt.gca().add_collection(lc)
  plt.xlim(-30, 30)
  plt.ylim(-30, 30)
  plt.show()
#  ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')

if __name__ == '__main__':
  root = os.environ['VOICE_VAE']

  # parse arguments
  args = parse_args()
  conf = parse_conf(root, args)

  test_2(conf, conf['test_dir'], '61-70968-0003.mfcc.npy')

  if args is None:
      exit()
