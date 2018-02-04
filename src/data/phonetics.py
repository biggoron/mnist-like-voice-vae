import os
from os import listdir
from os.path import join, isfile
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

ARPABET = {
  'h#':
    {
      'id': 0,
      '1_letter': '#',
      '2_letters': '##',
      'ipa': u"\u2205",
      'examples': ['...'],
      'color': 'whitesmoke',
      'category': 'non_semantic',
      'marker': '+'
    },
  'epi':
    {
      'id': 0,
      '1_letter': '#',
      '2_letters': '##',
      'ipa': u"\u2205",
      'examples': ['...'],
      'color': 'black',
      'category': 'non_semantic',
      'marker': '>'
    },
  'pau':
    {
      'id': 0,
      '1_letter': '#',
      '2_letters': '##',
      'ipa': u"\u2205",
      'examples': ['...'],
      'color': 'black',
      'category': 'non_semantic',
      'marker': 'o'
    },
  'aa':
    {
      'id': 6,
      '1_letter': 'a',
      '2_letters': 'aa',
      'ipa': u"\u0251",
      'examples': ['bALm', 'bOt'],
      'color': 'palevioletred',
      'category': 'vowel',
      'marker': 'o'
    },
  'ae':
    {
      'id': 5,
      '1_letter': '@',
      '2_letters': 'ae',
      'ipa': u'\u00E6',
      'examples': ['bAt'],
      'color': 'crimson',
      'category': 'vowel',
      'marker': 'o'
    },
  '@':
    {
      'id': 5,
      '1_letter': '@',
      '2_letters': 'ae',
      'ipa': u'\u00E6',
      'examples': ['bAt'],
      'color': 'crimson',
      'category': 'vowel',
      'marker': 'o'
    },
  'ah':
    {
      'id': 9,
      '1_letter': 'A',
      '2_letters': 'ah',
      'ipa': u"\u2038",
      'examples': ['bUtt'],
      'color': 'pink',
      'category': 'vowel',
      'marker': 'o'
    },
  'ao':
    {
      'id': 10,
      '1_letter': 'c',
      '2_letters': 'ao',
      'ipa': u"\u0063",
      'examples': ['bOUGHt'],
      'color': 'mediumvioletred',
      'category': 'vowel',
      'marker': 'o'
    },
  'aw':
    {
      'id': 7,
      '1_letter': 'W',
      '2_letters': 'aw',
      'ipa': u"\u0061\u028A",
      'examples': ['bOUt'],
      'color': 'deeppink',
      'category': 'vowel',
      'marker': 'o'
    },
  'ax-h':
    {
      'id': 20,
      '1_letter': 'x',
      '2_letters': 'ax',
      'ipa': u"\u0259",
      'examples': ['About'],
      'color': 'm',
      'category': 'vowel',
      'marker': '>'
    },
  'ax':
    {
      'id': 17,
      '1_letter': 'x',
      '2_letters': 'ax',
      'ipa': u"\u0259",
      'examples': ['About'],
      'color': 'm',
      'category': 'vowel',
      'marker': 'o'
    },
  'axr':
    {
      'id': 19,
      '1_letter': '2',
      '2_letters': 'axr',
      'ipa': u"\u025A",
      'examples': ['lettER'],
      'color': 'm',
      'category': 'vowel',
      'marker': '+'
    },
  'ay':
    {
      'id': 8,
      '1_letter': 'Y',
      '2_letters': 'ay',
      'ipa': u"\u0061\u026A",
      'examples': ['bIde'],
      'color': 'palevioletred',
      'category': 'vowel',
      'marker': '+'
    },
  'eh':
    {
      'id': 3,
      '1_letter': 'E',
      '2_letters': 'eh',
      'ipa': u"\u025B",
      'examples': ['bEt'],
      'color': 'purple',
      'category': 'vowel',
      'marker': 'o'
    },
  'er':
    {
      'id': 16,
      '1_letter': 'R',
      '2_letters': 'er',
      'ipa': u"\u025D",
      'examples': ['bIRd'],
      'color': 'pink',
      'category': 'vowel',
      'marker': '+'
    },
  'ey':
    {
      'id': 4,
      '1_letter': 'e',
      '2_letters': 'ey',
      'ipa': u"\u025B\u026A",
      'examples': ['bAIt'],
      'color': 'purple',
      'category': 'vowel',
      'marker': '+'
    },
  'ih':
    {
      'id': 2,
      '1_letter': 'I',
      '2_letters': 'ih',
      'ipa': u"\u026A",
      'examples': ['bIt'],
      'color': 'darkorchid',
      'category': 'vowel',
      'marker': 'o'
    },
  'ix':
    {
      'id': 18,
      '1_letter': 'X',
      '2_letters': 'ix',
      'ipa': u"\u0268",
      'examples': ['rosEs', 'rabbIt'],
      'color': 'darkviolet',
      'category': 'vowel',
      'marker': '>'
    },
  'iy':
    {
      'id': 1,
      '1_letter': 'i',
      '2_letters': 'iy',
      'ipa': u"\u0069",
      'examples': ['bEAt'],
      'color': 'mediumorchid',
      'category': 'vowel',
      'marker': '+'
    },
  'ow':
    {
      'id': 12,
      '1_letter': 'o',
      '2_letters': 'ow',
      'ipa': u"\u006F\u028A",
      'examples': ['bOAt'],
      'color': 'mediumvioletred',
      'category': 'vowel',
      'marker': '+'
    },
  'oy':
    {
      'id': 11,
      '1_letter': 'O',
      '2_letters': 'oy',
      'ipa': u"\u0254\u026A",
      'examples': ['boy'],
      'color': 'orchid',
      'category': 'vowel',
      'marker': '+'
    },
  'uh':
    {
      'id': 13,
      '1_letter': 'U',
      '2_letters': 'uh',
      'ipa': u"\u028A",
      'examples': ['book'],
      'color': 'blue',
      'category': 'vowel',
      'marker': 'o'
    },
  'uw':
    {
      'id': 14,
      '1_letter': 'u',
      '2_letters': 'uw',
      'ipa': u"\u0075",
      'examples': ['boot'],
      'color': 'slateblue',
      'category': 'vowel',
      'marker': 'o'
    },
  'ux':
    {
      'id': 15,
      '1_letter': '3',
      '2_letters': 'ux',
      'ipa': u"\u0289",
      'examples': ['dude'],
      'color': 'darkslateblue',
      'category': 'vowel',
      'marker': 'o'
    },
  'bcl':
    {
      'id': 46,
      '1_letter': 'b',
      '2_letters': 'b',
      'ipa': u"\u0062",
      'examples': ['Buy'],
      'color': 'rosybrown',
      'category': 'consonnant',
      'marker': '>'
    },
  'b':
    {
      'id': 45,
      '1_letter': 'b',
      '2_letters': 'b',
      'ipa': u"\u0062",
      'examples': ['Buy'],
      'color': 'rosybrown',
      'category': 'consonnant',
      'marker': 'o'
    },
  'ch':
    {
      'id': 44,
      '1_letter': 'C',
      '2_letters': 'ch',
      'ipa': u"\u0074\u0283",
      'examples': ['CHina'],
      'color': 'firebrick',
      'category': 'consonnant',
      'marker': 'o'
    },
  'dcl':
    {
      'id': 48,
      '1_letter': 'd',
      '2_letters': 'd',
      'ipa': u"\u0064",
      'examples': ['Die'],
      'color': 'red',
      'category': 'consonnant',
      'marker': '>'
    },
  'd':
    {
      'id': 47,
      '1_letter': 'd',
      '2_letters': 'd',
      'ipa': u"\u0064",
      'examples': ['Die'],
      'color': 'red',
      'category': 'consonnant',
      'marker': 'o'
    },
  'dh':
    {
      'id': 42,
      '1_letter': 'D',
      '2_letters': 'dh',
      'ipa': u"\u00F0",
      'examples': ['Thy'],
      'color': 'darksalmon',
      'category': 'consonnant',
      'marker': 'o'
    },
  'dx':
    {
      'id': 57,
      '1_letter': 'F',
      '2_letters': 'dx',
      'ipa': u"\u027E",
      'examples': ['buTTer'],
      'color': 'red',
      'category': 'consonnant',
      'marker': '^'
    },
  'el':
    {
      'id': 27,
      '1_letter': 'L',
      '2_letters': 'el',
      'ipa': u"\u006C\u0329",
      'examples': ['bottLE'],
      'color': 'sienna',
      'category': 'consonnant',
      'marker': '<'
    },
  'em':
    {
      'id': 31,
      '1_letter': 'M',
      '2_letters': 'em',
      'ipa': u"\u006D\u0329",
      'examples': ['rhythM'],
      'color': 'gold',
      'category': 'consonnant',
      'marker': '<'
    },
  'en':
    {
      'id': 32,
      '1_letter': 'N',
      '2_letters': 'en',
      'ipa': u"\u006E\u0329",
      'examples': ['buttON'],
      'color': 'darkkhaki',
      'category': 'consonnant',
      'marker': '<'
    },
  'f':
    {
      'id': 39,
      '1_letter': 'f',
      '2_letters': 'f',
      'ipa': u"\u0066",
      'examples': ['Fight'],
      'color': 'darkcyan',
      'category': 'consonnant',
      'marker': 'o'
    },
  'gcl':
    {
      'id': 50,
      '1_letter': 'g',
      '2_letters': 'g',
      'ipa': u"\u0067",
      'examples': ['Guy'],
      'color': 'seagreen',
      'category': 'consonnant',
      'marker': '>'
    },
    
  'g':
    {
      'id': 49,
      '1_letter': 'g',
      '2_letters': 'g',
      'ipa': u"\u0067",
      'examples': ['Guy'],
      'color': 'seagreen',
      'category': 'consonnant',
      'marker': 'o'
    },
  'hv':
    {
      'id': 26,
      '1_letter': 'h',
      '2_letters': 'hv',
      'ipa': u"\u0068",
      'examples': ['High'],
      'color': 'palegreen',
      'category': 'consonnant',
      'marker': '>'
    },
  'hh':
    {
      'id': 25,
      '1_letter': 'h',
      '2_letters': 'hh',
      'ipa': u"\u0068",
      'examples': ['High'],
      'color': 'palegreen',
      'category': 'consonnant',
      'marker': 'o'
    },
  'jh':
    {
      'id': 43,
      '1_letter': 'J',
      '2_letters': 'jh',
      'ipa': u"\u0064\u0292",
      'examples': ['Jive'],
      'color': 'lawngreen',
      'category': 'consonnant',
      'marker': 'o'
    },
  'kcl':
    {
      'id': 56,
      '1_letter': 'k',
      '2_letters': 'k',
      'ipa': u"\u006B",
      'examples': ['Kite'],
      'color': 'deepskyblue',
      'category': 'consonnant',
      'marker': '>'
    },
  'k':
    {
      'id': 55,
      '1_letter': 'k',
      '2_letters': 'k',
      'ipa': u"\u006B",
      'examples': ['Kite'],
      'color': 'deepskyblue',
      'category': 'consonnant',
      'marker': 'o'
    },
  'l':
    {
      'id': 21,
      '1_letter': 'l',
      '2_letters': 'l',
      'ipa': u"\u006C",
      'examples': ['Lie'],
      'color': 'sienna',
      'category': 'consonnant',
      'marker': 'o'
    },
  'm':
    {
      'id': 28,
      '1_letter': 'm',
      '2_letters': 'm',
      'ipa': u"\u006D",
      'examples': ['My'],
      'color': 'gold',
      'category': 'consonnant',
      'marker': 'o'
    },
  'n':
    {
      'id': 29,
      '1_letter': 'n',
      '2_letters': 'n',
      'ipa': u"\u006E",
      'examples': ['Nigh'],
      'color': 'darkkhaki',
      'category': 'consonnant',
      'marker': 'o'
    },
  'eng':
    {
      'id': 33,
      '1_letter': 'G',
      '2_letters': 'eng',
      'ipa': u"\u014B",
      'examples': ['siNG'],
      'color': 'y',
      'category': 'consonnant',
      'marker': '>'
    },
  'ng':
    {
      'id': 30,
      '1_letter': 'G',
      '2_letters': 'ng',
      'ipa': u"\u014B",
      'examples': ['siNG'],
      'color': 'y',
      'category': 'consonnant',
      'marker': 'o'
    },
  'nx':
    {
      'id': 34,
      '1_letter': '1',
      '2_letters': 'nx',
      'ipa': u"\u027E\u0303",
      'examples': ['wiNNer'],
      'color': 'y',
      'category': 'consonnant',
      'marker': '^'
    },
  'pcl':
    {
      'id': 52,
      '1_letter': 'p',
      '2_letters': 'p',
      'ipa': u"\u0070",
      'examples': ['Pie'],
      'color': 'aqua',
      'category': 'consonnant',
      'marker': '>'
    },
    
  'p':
    {
      'id': 51,
      '1_letter': 'p',
      '2_letters': 'p',
      'ipa': u"\u0070",
      'examples': ['Pie'],
      'color': 'aqua',
      'category': 'consonnant',
      'marker': 'o'
    },
  'q':
    {
      'id': 58,
      '1_letter': 'Q',
      '2_letters': 'q',
      'ipa': u"\u0294",
      'examples': ['uh>-<oh'],
      'color': 'peachpuff',
      'category': 'consonnant',
      'marker': 'o'
    },
  'r':
    {
      'id': 22,
      '1_letter': 'r',
      '2_letters': 'r',
      'ipa': u"\u0279",
      'examples': ['Rye'],
      'color': 'g',
      'category': 'consonnant',
      'marker': 'o'
    },
  's':
    {
      'id': 35,
      '1_letter': 's',
      '2_letters': 's',
      'ipa': u"\u0073",
      'examples': ['Sigh'],
      'color': 'lime',
      'category': 'consonnant',
      'marker': 'o'
    },
  'sh':
    {
      'id': 36,
      '1_letter': 'S',
      '2_letters': 'sh',
      'ipa': u"\u0283",
      'examples': ['SHy'],
      'color': 'limegreen',
      'category': 'consonnant',
      'marker': 'o'
    },
  'tcl':
    {
      'id': 54,
      '1_letter': 't',
      '2_letters': 't',
      'ipa': u"\u0074",
      'examples': ['Tie'],
      'color': 'wheat',
      'category': 'consonnant',
      'marker': '>',
    },
    
  't':
    {
      'id': 53,
      '1_letter': 't',
      '2_letters': 't',
      'ipa': u"\u0074",
      'examples': ['Tie'],
      'color': 'wheat',
      'category': 'consonnant',
      'marker': 'o'
    },
  'th':
    {
      'id': 40,
      '1_letter': 'T',
      '2_letters': 'th',
      'ipa': u"\u03B8",
      'examples': ['THigh'],
      'color': 'coral',
      'category': 'consonnant',
      'marker': 'o'
    },
  'v':
    {
      'id': 41,
      '1_letter': 'v',
      '2_letters': 'v',
      'ipa': u"\u0076",
      'examples': ['Vie'],
      'color': 'c',
      'category': 'consonnant',
      'marker': 'o'
    },
  'w':
    {
      'id': 23,
      '1_letter': 'w',
      '2_letters': 'w',
      'ipa': u"\u0077",
      'examples': ['Wise'],
      'color': 'bisque',
      'category': 'consonnant',
      'marker': 'o'
    },
  'wh':
    {
      'id': 23,
      '1_letter': 'H',
      '2_letters': 'wh',
      'ipa': u"\u028D",
      'examples': ['WHy'],
      'color': 'sandybrown',
      'category': 'consonnant',
      'marker': 'o'
    },
  'y':
    {
      'id': 24,
      '1_letter': 'y',
      '2_letters': 'y',
      'ipa': u"\u006A",
      'examples': ['Yacht'],
      'color': 'powderblue',
      'category': 'consonnant',
      'marker': 'o'
    },
  'z':
    {
      'id': 37,
      '1_letter': 'z',
      '2_letters': 'z',
      'ipa': u"\u007A",
      'examples': ['Zoo'],
      'color': 'salmon',
      'category': 'consonnant',
      'marker': 'o'
    },
  'zh':
    {
      'id': 38,
      '1_letter': 'Z',
      '2_letters': 'zh',
      'ipa': u"\u0292",
      'examples': ['pleaSure'],
      'color': 'orange',
      'category': 'consonnant',
      'marker': 'o'
    }
}
def phn_to_index(phn) :
  return ARPABET[phn]['id']
def index_to_phn(i):
  if i == -1:
    return '_'
  for phn, v in ARPABET.items():
    if v['id'] == i:
      return phn
  

class PhonemeTable():
  def __init__(self, phn_dir, phn_name):
    phn_file = "%s.phn" % phn_name
    txt_file = "%s.txt" % phn_name
    phn_path = os.path.join(phn_dir, phn_file)
    txt_path = os.path.join(phn_dir, txt_file)
    with open(txt_path) as f:
      content = f.readlines()
      self.duration = int(content[0].strip().split(' ')[1])
    with open(phn_path) as f:
      content = f.readlines()
    content = [x.strip() for x in content] 
    content = [x.split(' ') for x in content]
    self.content = [{
                      'phn': x[2],
                      'beg': int(x[0]) / float(self.duration),
                      'end': int(x[1]) / float(self.duration)
                    } for x in content]

  def phoneme_at(self, s):
    i = 0
    while self.content[i]['end'] < s and i < len(self.content) - 1:
      i += 1
    return self.content[i]['phn']

  def color_of(self, phn):
    return 'red'

class WordTable():
  def __init__(self, word_dir, word_name):
    word_file = "%s.wrd" % word_name
    txt_file = "%s.txt" % word_name
    word_path = os.path.join(word_dir, word_file)
    txt_path = os.path.join(word_dir, txt_file)
    with open(txt_path) as f:
      content = f.readlines()
      self.duration = int(content[0].strip().split(' ')[1])
    with open(word_path) as f:
      content = f.readlines()
    content = [x.strip() for x in content] 
    content = [x.split(' ') for x in content]
    self.content = [{
                      'wrd': x[2],
                      'beg': int(x[0]) / float(self.duration),
                      'end': int(x[1]) / float(self.duration)
                    } for x in content]

  def word_at(self, s):
    i = 0
    while self.content[i]['end'] < s and i < len(self.content) - 1:
      i += 1
    return self.content[i]['wrd']

if __name__ == '__main__':
#  for pho, v in ARPABET.items():
#    print('%s :  %s' % (pho, v['ipa']))
#    fig, ax = plt.subplots()
#    ax.set_facecolor(v['color'])
#    plt.show()
#    plt.close()
  data_dir = 'data/timit_data/'
  ft = [
    join(data_dir, f)
    for f in listdir(data_dir)
    if isfile(join(data_dir, f)) and f[-4:] == '.phn'
  ]
  paths = [os.path.split(fn) for fn in ft]
  phonemes = [PhonemeTable(p[0], p[1][:-4]) for p in paths]

  for pf in phonemes:
    for i in pf.content:
      k = i['phn']
      if k not in ARPABET.keys():
        print(k)
      
    

