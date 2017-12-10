import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

ARPABET = {
  'aa':
    {
      '1_letter': 'a',
      '2_letters': 'aa',
      'ipa': u"\u0251",
      'examples': ['bALm', 'bOt'],
      'color': 'palevioletred',
      'category': 'vowel'
    },
  '@':
    {
      '1_letter': '@',
      '2_letters': 'ae',
      'ipa': u'\u00E6',
      'examples': ['bAt'],
      'color': 'crimson',
      'category': 'vowel'
    },
  'ah':
    {
      '1_letter': 'A',
      '2_letters': 'ah',
      'ipa': u"\u2038",
      'examples': ['bUtt'],
      'color': 'pink',
      'category': 'vowel'
    },
  'ao':
    {
      '1_letter': 'c',
      '2_letters': 'ao',
      'ipa': u"\u0063",
      'examples': ['bOUGHt'],
      'color': 'mediumvioletred',
      'category': 'vowel'
    },
  'aw':
    {
      '1_letter': 'W',
      '2_letters': 'aw',
      'ipa': u"\u0061\u028A",
      'examples': ['bOUt'],
      'color': 'deeppink',
      'category': 'vowel'
    },
  'ax':
    {
      '1_letter': 'x',
      '2_letters': 'ax',
      'ipa': u"\u0259",
      'examples': ['About'],
      'color': 'm',
      'category': 'vowel'
    },
  'axr':
    {
      '1_letter': None,
      '2_letters': 'axr',
      'ipa': u"\u025A",
      'examples': ['lettER'],
      'color': 'silver',
      'category': 'vowel'
    },
  'ay':
    {
      '1_letter': 'Y',
      '2_letters': 'ay',
      'ipa': u"\u0061\u026A",
      'examples': ['bIde'],
      'color': 'plum',
      'category': 'vowel'
    },
  'eh':
    {
      '1_letter': 'E',
      '2_letters': 'eh',
      'ipa': u"\u025B",
      'examples': ['bEt'],
      'color': 'purple',
      'category': 'vowel'
    },
  'er':
    {
      '1_letter': 'R',
      '2_letters': 'er',
      'ipa': u"\u025D",
      'examples': ['bIRd'],
      'color': 'gray',
      'category': 'vowel'
    },
  'ey':
    {
      '1_letter': 'e',
      '2_letters': 'ey',
      'ipa': u"\u025B\u026A",
      'examples': ['bAIt'],
      'color': 'magenta',
      'category': 'vowel'
    },
  'ih':
    {
      '1_letter': 'I',
      '2_letters': 'ih',
      'ipa': u"\u026A",
      'examples': ['bIt'],
      'color': 'darkorchid',
      'category': 'vowel'
    },
  'ix':
    {
      '1_letter': 'X',
      '2_letters': 'ix',
      'ipa': u"\u0268",
      'examples': ['rosEs', 'rabbIt'],
      'color': 'darkviolet',
      'category': 'vowel'
    },
  'iy':
    {
      '1_letter': 'i',
      '2_letters': 'iy',
      'ipa': u"\u0069",
      'examples': ['bEAt'],
      'color': 'mediumorchid',
      'category': 'vowel'
    },
  'ow':
    {
      '1_letter': 'o',
      '2_letters': 'ow',
      'ipa': u"\u006F\u028A",
      'examples': ['bOAt'],
      'color': 'lavenderblush',
      'category': 'vowel'
    },
  'oy':
    {
      '1_letter': 'O',
      '2_letters': 'oy',
      'ipa': u"\u0254\u026A",
      'examples': ['boy'],
      'color': 'orchid',
      'category': 'vowel'
    },
  'uh':
    {
      '1_letter': 'U',
      '2_letters': 'uh',
      'ipa': u"\u028A",
      'examples': ['book'],
      'color': 'blue',
      'category': 'vowel'
    },
  'uw':
    {
      '1_letter': 'u',
      '2_letters': 'uw',
      'ipa': u"\u0075",
      'examples': ['boot'],
      'color': 'slateblue',
      'category': 'vowel'
    },
  'ux':
    {
      '1_letter': None,
      '2_letters': 'ux',
      'ipa': u"\u0289",
      'examples': ['dude'],
      'color': 'darkslateblue',
      'category': 'vowel'
    },
  'b':
    {
      '1_letter': 'b',
      '2_letters': 'b',
      'ipa': u"\u0062",
      'examples': ['Buy'],
      'color': 'rosybrown',
      'category': 'consonnant'
    },
  'ch':
    {
      '1_letter': 'C',
      '2_letters': 'ch',
      'ipa': u"\u0074\u0283",
      'examples': ['CHina'],
      'color': 'firebrick',
      'category': 'consonnant'
    },
  'd':
    {
      '1_letter': 'd',
      '2_letters': 'd',
      'ipa': u"\u0064",
      'examples': ['Die'],
      'color': 'red',
      'category': 'consonnant'
    },
  'dh':
    {
      '1_letter': 'D',
      '2_letters': 'dh',
      'ipa': u"\u00F0",
      'examples': ['Thy'],
      'color': 'darksalmon',
      'category': 'consonnant'
    },
  'dx':
    {
      '1_letter': 'F',
      '2_letters': 'dx',
      'ipa': u"\u027E",
      'examples': ['buTTer'],
      'color': 'orangered',
      'category': 'consonnant'
    },
  'el':
    {
      '1_letter': 'L',
      '2_letters': 'el',
      'ipa': u"\u006C\u0329",
      'examples': ['bottLE'],
      'color': 'sienna',
      'category': 'consonnant'
    },
  'em':
    {
      '1_letter': 'M',
      '2_letters': 'em',
      'ipa': u"\u006D\u0329",
      'examples': ['rhythM'],
      'color': 'gold',
      'category': 'consonnant'
    },
  'en':
    {
      '1_letter': 'N',
      '2_letters': 'en',
      'ipa': u"\u006E\u0329",
      'examples': ['buttON'],
      'color': 'darkkhaki',
      'category': 'consonnant'
    },
  'f':
    {
      '1_letter': 'f',
      '2_letters': 'f',
      'ipa': u"\u0066",
      'examples': ['Fight'],
      'color': 'darkcyan',
      'category': 'consonnant'
    },
  'g':
    {
      '1_letter': 'g',
      '2_letters': 'g',
      'ipa': u"\u0067",
      'examples': ['Guy'],
      'color': 'seagreen',
      'category': 'consonnant'
    },
  'hh':
    {
      '1_letter': 'h',
      '2_letters': 'hh',
      'ipa': u"\u0068",
      'examples': ['High'],
      'color': 'palegreen',
      'category': 'consonnant'
    },
  'jh':
    {
      '1_letter': 'J',
      '2_letters': 'jh',
      'ipa': u"\u0064\u0292",
      'examples': ['Jive'],
      'color': 'lawngreen',
      'category': 'consonnant'
    },
  'k':
    {
      '1_letter': 'k',
      '2_letters': 'k',
      'ipa': u"\u006B",
      'examples': ['Kite'],
      'color': 'deepskyblue',
      'category': 'consonnant'
    },
  'l':
    {
      '1_letter': 'l',
      '2_letters': 'l',
      'ipa': u"\u006C",
      'examples': ['Lie'],
      'color': 'chocolate',
      'category': 'consonnant'
    },
  'm':
    {
      '1_letter': 'm',
      '2_letters': 'm',
      'ipa': u"\u006D",
      'examples': ['My'],
      'color': 'yellow',
      'category': 'consonnant'
    },
  'n':
    {
      '1_letter': 'n',
      '2_letters': 'n',
      'ipa': u"\u006E",
      'examples': ['Nigh'],
      'color': 'olive',
      'category': 'consonnant'
    },
  'ng':
    {
      '1_letter': 'G',
      '2_letters': 'ng',
      'ipa': u"\u014B",
      'examples': ['siNG'],
      'color': 'y',
      'category': 'consonnant'
    },
  'nx':
    {
      '1_letter': None,
      '2_letters': 'nx',
      'ipa': u"\u027E\u0303",
      'examples': ['wiNNer'],
      'color': 'olivedrab',
      'category': 'consonnant'
    },
  'p':
    {
      '1_letter': 'p',
      '2_letters': 'p',
      'ipa': u"\u0070",
      'examples': ['Pie'],
      'color': 'aqua',
      'category': 'consonnant'
    },
  'q':
    {
      '1_letter': 'Q',
      '2_letters': 'q',
      'ipa': u"\u0294",
      'examples': ['uh>-<oh'],
      'color': 'peachpuff',
      'category': 'consonnant'
    },
  'r':
    {
      '1_letter': 'r',
      '2_letters': 'r',
      'ipa': u"\u0279",
      'examples': ['Rye'],
      'color': 'whitesmoke',
      'category': 'consonnant'
    },
  's':
    {
      '1_letter': 's',
      '2_letters': 's',
      'ipa': u"\u0073",
      'examples': ['Sigh'],
      'color': 'lime',
      'category': 'consonnant'
    },
  'sh':
    {
      '1_letter': 'S',
      '2_letters': 'sh',
      'ipa': u"\u0283",
      'examples': ['SHy'],
      'color': 'limegreen',
      'category': 'consonnant'
    },
  't':
    {
      '1_letter': 't',
      '2_letters': 't',
      'ipa': u"\u0074",
      'examples': ['Tie'],
      'color': 'wheat',
      'category': 'consonnant'
    },
  'th':
    {
      '1_letter': 'T',
      '2_letters': 'th',
      'ipa': u"\u03B8",
      'examples': ['THigh'],
      'color': 'coral',
      'category': 'consonnant'
    },
  'v':
    {
      '1_letter': 'v',
      '2_letters': 'v',
      'ipa': u"\u0076",
      'examples': ['Vie'],
      'color': 'c',
      'category': 'consonnant'
    },
  'w':
    {
      '1_letter': 'w',
      '2_letters': 'w',
      'ipa': u"\u0077",
      'examples': ['Wise'],
      'color': 'bisque',
      'category': 'consonnant'
    },
  'wh':
    {
      '1_letter': 'H',
      '2_letters': 'wh',
      'ipa': u"\u028D",
      'examples': ['WHy'],
      'color': 'sandybrown',
      'category': 'consonnant'
    },
  'y':
    {
      '1_letter': 'y',
      '2_letters': 'y',
      'ipa': u"\u006A",
      'examples': ['Yacht'],
      'color': 'powderblue',
      'category': 'consonnant'
    },
  'z':
    {
      '1_letter': 'z',
      '2_letters': 'z',
      'ipa': u"\u007A",
      'examples': ['Zoo'],
      'color': 'salmon',
      'category': 'consonnant'
    },
  'zh':
    {
      '1_letter': 'Z',
      '2_letters': 'zh',
      'ipa': u"\u0292",
      'examples': ['pleaSure'],
      'color': 'orange',
      'category': 'consonnant'
    }
}

if __name__ == '__main__':
  for pho, v in ARPABET.items():
    print('%s :  %s' % (pho, v['ipa']))
    fig, ax = plt.subplots()
    ax.set_facecolor(v['color'])
    plt.show()
    plt.close()

