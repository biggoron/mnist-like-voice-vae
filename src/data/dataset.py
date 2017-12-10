import numpy as np
import os
import logging
from os import listdir
from os.path import isfile, join

class VoiceData():
  def __init__(self, config):
    print('loading init')
    self.load_config(config)
    print('finished loading conf')
    self.train_file_table = [
      join(self.train_dir, f)
      for f in listdir(self.train_dir)
      if isfile(join(self.train_dir, f)) and f[-8:] == 'mfcc.npy'
    ]
    print('built train ft')
    self.test_file_table = [
      join(self.test_dir, f)
      for f in listdir(self.test_dir)
      if isfile(join(self.test_dir, f)) and f[-8:] == 'mfcc.npy'
    ]
    print('built test ft')
    np.random.shuffle(self.train_file_table)
    self.train = Dataset(config, self.train_file_table, 'train')
    print('built train')
    self.test = Dataset(config, self.test_file_table, 'test')
    print('built test')

  def load_config(self, config):
    self.train_dir = config['train_dir']
    self.dev_dir = config['dev_dir']
    self.test_dir = config['test_dir']


class Dataset():
  def __init__(self, config, ft, phase):
    print('  init %s' % phase)
    self.phase = phase
    self.load_config(config)
    print('  loaded conf %s' % phase)
    self.ft = ft[:self.nb_files]
    print('  cut length %s' % phase)
    self.data = np.asarray([np.load(fn) for fn in self.ft])
    print('  build data %s' % phase)
    self.ls = [len(f) for f in self.data]
    print('  build len %s' % phase)

    self.elem_nb = 0
    for l in self.ls:
      self.elem_nb += (l - 12)
    print('  build elem nb %s' % phase)

    self.elem_ids = []
    for i in range(len(self.data)):
      for j in range(self.ls[i] - 12):
        self.elem_ids.append([i, j])
    print('  build elem ids %s' % phase)
    np.random.shuffle(self.elem_ids)

    self.i = 0

  def shuffle_data(self):
    np.random.shuffle(self.elem_ids)

  def reset_index(self, j=0):
    self.i = j

  def load_config(self, config):
    self.nb_files = config['%s_nb' % self.phase]

  def next_batch(self, batch_size, offset = None):
    if offset == None:
      offset = i
    b = []
    for i in range(batch_size):
      file_id, pos_id = self.elem_ids[offset]
      elem = self.data[file_id][pos_id:pos_id+13]
      b.append(elem)
      offset += 1
    ret = np.reshape(np.asarray(b), [batch_size, 13*13])
    ret = ret - np.mean(ret, axis = 0)
    ret = ret / np.std(ret, axis=0)
    return  ret

class FileData():
  def __init__(self, test_dir, fn):
    self.phase = 'test'
    fn = join(test_dir, fn)
    self.data = np.asarray(np.load(fn))
    self.l = len(self.data)
    self.elem_nb = self.l - 12
    self.elem_ids = list(range(self.elem_nb))
    self.i = 0

  def reset_index(self, j=0):
    self.i = j

  def next_batch(self, batch_size, offset = None):
    if offset == None:
      offset = i
    b = []
    for i in range(batch_size):
      pos_id = self.elem_ids[offset]
      elem = self.data[pos_id:pos_id+13]
      b.append(elem)
      offset += 1
    ret = np.reshape(np.asarray(b), [batch_size, 13*13])
    ret = ret - np.mean(ret, axis = 0)
    ret = ret / np.std(ret, axis=0)
    return  ret
