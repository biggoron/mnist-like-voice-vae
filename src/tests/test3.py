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
import src.data.dataset2 as dt
import src.graphs.graph as gr

import src.data.phonetics as ph

def test_3(config, data_files):
  batch_size = config['batch_size']

  arpabet = ph.ARPABET

  tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

  graph = gr.Graph('saved_models/predict/', 'save_26.ckpt.meta')

  datastore = dt.DataStore()
  datastore.create_collection('one_file')
  for data_file in data_files:
    datastore.file_to_collection(data_file, 'one_file')
  dataset = datastore.dataset('one_file', step = 1, ftnb = 29, width = 26)
  n_samples = dataset.elem_nb

  total_batch = int(n_samples / conf['batch_size'])
  tbar = tqdm(range(total_batch))

  points, labels = graph.embed(dataset, batch_size, with_labels = True)

  l_tsne = tsne.fit_transform(points)
  x = l_tsne[:256,0]
  y = l_tsne[:256,1]
  default = {'color': 'black', '2_letters': 'None', 'marker': 'o'}
  phn = [d['phn'] for d in labels[:256]]
  phn = [arpabet.get(p, default) for p in phn]
  c = [p['color'] for p in phn]
  txt = ['$%s$' % p['2_letters'] for p in phn]
  mrq = [p['marker'] for p in phn]


  points = np.array([x, y]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  l = len(segments)
  t = np.linspace(0, 1, l)
  lc = LineCollection(segments, linewidth=0.1, cmap = plt.get_cmap('viridis'), linestyle='solid')
  lc.set_array(t)
#  lc.set_linewidth(2)
  
  fig = plt.figure(figsize = (20, 20), dpi = 500)

  plt.gca().add_collection(lc)
  for i in range(len(x)):
    plt.scatter(x[i], y[i], c=c[i], s = 12, marker=mrq[i], edgecolors=c[i], linewidths=0.1)

  plt.xlim(-80, 80)
  plt.ylim(-80, 80)
  plt.show()

if __name__ == '__main__':
  root = os.environ['VOICE_VAE']

  # parse arguments
  args = parse_args()
  conf = parse_conf(root, args)

  data_dir = 'data/timit_data/'
  data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and f[-4:] == '.npy' ][0:5]

  test_3(conf, data_files)

  if args is None:
      exit()

