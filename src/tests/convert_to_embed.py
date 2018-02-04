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
  batch_size = 16

  graph = gr.Graph('saved_models/predict/', 'save_26.ckpt.meta')

  tbar = tqdm(data_files)
  for data_file in tbar:
    datastore = dt.DataStore()
    datastore.create_collection(data_file)
    datastore.file_to_collection(data_file, data_file)
    dataset = datastore.dataset(data_file, labels=False, step = 1, ftnb = 29, width = 26)
    n_samples = dataset.elem_nb
    total_batch = int(n_samples)
    points = graph.embed(dataset, batch_size, with_labels = False)
    np.save("%s.emb" % data_file[:-4], points)
    datastore.create_collection(data_file)
    

if __name__ == '__main__':
  root = os.environ['VOICE_VAE']

  # parse arguments
  args = parse_args()
  conf = parse_conf(root, args)

  data_dir = 'data/timit_data/'
  data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and f[-4:] == '.npy']

  test_3(conf, data_files)

  if args is None:
      exit()

