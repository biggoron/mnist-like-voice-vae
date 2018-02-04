import os
import pickle
from matplotlib.collections import LineCollection

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from sklearn import manifold

from src.utils.load_conf import parse_args, parse_conf
import src.data.dataset as dt
import src.graphs.graph as gr

def test_1(config):
  dataset = dt.VoiceData(config)
  n_samples = dataset.test.elem_nb
  batch_size = config['batch_size']


  graph = gr.Graph('saved_models/complicated_no_distr/', 'save_3.ckpt.meta')

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  points = graph.embed(dataset.test, batch_size)

  tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
  l_tsne = tsne.fit_transform(points)

  ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
  plt.show()


if __name__ == '__main__':
  root = os.environ['VOICE_VAE']

  # parse arguments
  args = parse_args()
  conf = parse_conf(root, args)

  test_1(conf, conf['test_dir'], '61-70968-0003.mfcc.npy')

  if args is None:
      exit()
