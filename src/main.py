import os

from utils.load_conf import parse_args, parse_conf
from run_main import run_mnist, run_voice

if __name__ == '__main__':
  root = os.environ['VOICE_VAE']
  # parse arguments
  args = parse_args()
  conf = parse_conf(root, args)

  if args is None:
      exit()

  # main
  run_voice(conf)
