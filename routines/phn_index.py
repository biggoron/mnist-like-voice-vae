from src.utils.text import phn_to_char_array, normalize_phn_file
import src.data.phonetics as ph
import os
from os import listdir
from os.path import isfile, join

d = 'data/timit_test/'
phn_files = [join(d, f) for f in listdir(d) if isfile(join(d, f)) and '.phn' in f]
txts = [phn_to_char_array(normalize_phn_file(fn)) for fn in phn_files]
txts2 = []
for txt in txts:
  txt = [ph.index_to_phn(c) for c in txt]
  txts2.append(' '.join(txt))

with open('phn.log', 'w') as f:
  for t, p in zip(phn_files, txts2):
    print('', file=f)
    print(t, file=f)
    print(p, file=f)
    print('', file=f)
  
