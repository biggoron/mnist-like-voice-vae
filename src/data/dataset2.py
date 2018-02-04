import numpy as np
import sys
import os
import logging
from os import listdir
from os.path import isfile, join
from src.utils.load_audio_to_mem import pad_sequences
from src.data.phonetics import PhonemeTable, WordTable 
from src.utils.text import sparse_tuple_from, text_to_char_array, normalize_txt_file, phn_to_char_array, normalize_phn_file

class DataStore():
  def __init__(self):
    self.collections = {'all': []}

  def create_collection(self, nm):
    self.collections[nm] = []

  def del_collection(self, nm):
    del self.collections[nm]

  def dir_to_collection(self, data_dir, col, emb = False, txt_phn = False):
    if col != 'all':
      self.dir_to_collection(data_dir, 'all')
    if emb:
      ft = [
        join(data_dir, f)
        for f in listdir(data_dir)
        if isfile(join(data_dir, f)) and f[-8:] == '.emb.npy'
      ]
    else:
      ft = [
        join(data_dir, f)
        for f in listdir(data_dir)
        if isfile(join(data_dir, f)) and f[-4:] == '.npy' and f[-8:-4] != '.emb'
      ]
    for fn in ft:
      self.collections[col].append(fn)

  def file_to_collection(self, data_path, col):
    if col != 'all':
      self.file_to_collection(data_path, 'all')
    if isfile(data_path) and data_path[-4:] == '.npy':
      self.collections[col].append(data_path)

  def shuffle_collection(self, nm):
    np.random.shuffle(self.collections[nm])
  
  def dataset(self, nm, labels=True, step=1, ftnb = 25, width=13):
    return Dataset(self.collections[nm], labels = labels, step = step, ftnb = ftnb, width = width)

  def dataset_rnn(self, nm, step=1, ftnb = 25, width=13, txt_phn = False):
    return DatasetRNN(self.collections[nm], step = step, ftnb = ftnb, width = width, txt_phn = txt_phn)

class Dataset():
  def __init__(self, ft, max_size = False, labels = True, step = 1, ftnb = 13, width = 13, txt_phn = False):
    self.labels = labels
    self.ftnb = ftnb
    self.step = step
    self.width = width
    self.txt_phn = txt_phn
    if max_size:
      self.ft = ft[:max_size]
    else:
      self.ft = ft
    self.data = np.asarray([np.load(fn) for fn in self.ft])
    for i in range(len(self.data)):
      j = len(self.data) - (i + 1)
      for k in self.data[j]:
        if len(k) != 29:
          np.delete(self.data, j, 0)
    self.ls = [len(f) for f in self.data]
    self.paths = [os.path.split(fn) for fn in self.ft]
    if self.labels:
      self.phonemes = np.asarray([PhonemeTable(p[0], p[1][:-4]) for p in self.paths])
      self.words = np.asarray([WordTable(p[0], p[1][:-4]) for p in self.paths])
      self.speakers = np.asarray([os.path.basename(fn).split('_')[0] for fn in self.ft])
      self.accents = np.asarray([os.path.basename(fn).split('_')[1] for fn in self.ft])
      self.sentences = np.asarray([os.path.basename(fn).split('_')[4] for fn in self.ft])

    self.elem_nb = 0
    for l in self.ls:
      self.elem_nb += int(float(l - (self.width)) / float(self.step)) + 1

    self.elem_ids = []
    for i in range(len(self.data)):
      for j in range(0, self.ls[i] - (self.width -1), self.step):
        self.elem_ids.append([i, j])

    self.i = 0

  def shuffle_data(self):
    np.random.shuffle(self.elem_ids)

  def reset_index(self, j=0):
    self.i = j

  def next_batch(self, batch_size, offset = None, dropout=False, fulls=False):
    if offset == None:
      offset = self.i
    b = []
    fts = []
    for i in range(batch_size):
      file_id, pos_id = self.elem_ids[offset]
      elem = self.data[file_id][pos_id:pos_id + self.width, :self.ftnb]
      if elem.shape[1] < self.ftnb:
        elem = np.zeros((self.width, self.ftnb))
      if self.labels:
        r = (pos_id + 5.) / self.ls[file_id]
        phn = self.phonemes[file_id].phoneme_at(r)
        wrd = self.words[file_id].word_at(r)
        fts.append({
          'phn': phn,
          'wrd': wrd,
          'spk': self.speakers[file_id],
          'acc': self.accents[file_id],
          'stc': self.sentences[file_id]
        })
      if fulls and np.random.randint(0, 256) == 0:
        elem = np.ones_like(elem)
      if fulls and np.random.randint(0, 256) == 0:
        p = np.random.randint(0, 10)
        q = 10 - p 
        elems = np.random.choice([0, 1], size=(self.width, self.ftnb), p=[p/10., q/10.])
      if dropout and np.random.randint(0, 256) == 0:
        elem = np.zeros_like(elem)
      b.append(elem)
      offset += 1
    ret = np.reshape(np.asarray(b), [batch_size, self.ftnb*self.width])
    ret = ret - np.mean(ret, axis = 0)
    ret = ret / np.std(ret, axis=0)
    if self.labels:
      return  ret, np.asarray(fts)
    else:
      return ret

class DatasetRNN():
  def __init__(self, ft, max_size = False, step = 1, ftnb = 13, width = 9, txt_phn = False):
    self.ftnb = ftnb
    self.step = step
    self.width = width
    self.txt_phn = txt_phn
    if max_size:
      self.ft = ft[:max_size]
    else:
      self.ft = ft
    if not self.txt_phn:
      self._txt_files = np.asarray([fn.replace('.emb.npy', '.txt').replace('.npy', '.txt') for fn in self.ft])
    else :
      self._txt_files = np.asarray([fn.replace('.emb.npy', '.phn').replace('.npy', '.phn') for fn in self.ft])
    self.data = np.asarray([np.load(fn) for fn in self.ft])
    for i in range(len(self.data)):
      j = len(self.data) - (i + 1)
      if self.data[j].shape[1] < self.ftnb:
        np.delete(self.data, j, 0)
    self.l = len(self.data)
    self.ls = [len(f) for f in self.data]
    self.paths = [os.path.split(fn) for fn in self.ft]
    self.elem_nb = self.l
    self.elem_ids = []
    for i in range(len(self.data)):
      self.elem_ids.append(i)
    self.i = 0

  def shuffle_data(self):
    np.random.shuffle(self.elem_ids)

  def reset_index(self, j=0):
    self.i = j

  def next_batch(self, batch_size, offset = None):
    if offset == None:
      offset = self.i
    try:
      assert offset < self.l
    except:
      print('offset should be smaller than file nb')
    b = []
    txts = []
    for i in range(batch_size):
      file_id = self.elem_ids[offset]
      txts.append(self._txt_files[file_id])
      try: 
        assert len(self._txt_files[file_id]) > 0 
      except:
        print('txt fn should be nn nul str')
      dt = self.data[file_id]
#      dt = dt - np.mean(dt, axis=0)
#      dt = dt / np.std(dt, axis=0)
      try: 
        assert dt.shape[1] >= self.ftnb
      except:
        print(self._txt_files[file_id])
        print(self._txt_files[file_id])
        print(self._txt_files[file_id])
        print(self._txt_files[file_id])
        print('dt should have ftnb values at least')
      elems = []
      n = 2*self.width + 1
      seq_l = int((self.ls[file_id] + 1 - n) / self.step)
      for j in range(seq_l):
        pos_id = j * self.step
        elems.append(dt[pos_id:pos_id + n, :self.ftnb])
      elems = np.reshape(np.asarray(elems), [-1, self.ftnb * (2*self.width + 1)])
      b.append(elems)
      offset += 1
      offset = offset % self.l
    self.i = offset
    b = np.asarray(b)
    b, seq_ls = pad_sequences(b)
    if not self.txt_phn:
      txts = np.asarray([text_to_char_array(normalize_txt_file(fn)) for fn in txts])
    else:
      txts = np.asarray([phn_to_char_array(normalize_phn_file(fn)) for fn in txts])
    txts = sparse_tuple_from(txts)
    return b, seq_ls, txts
