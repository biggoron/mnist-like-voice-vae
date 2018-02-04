import os

from utils.load_conf import parse_args, parse_conf
from utils.dirs import make_dir, clear_dirs
from run_main import run_timit

if __name__ == '__main__':
  # Locate the root directory of the project
  root = os.environ['VOICE_VAE']

  # parse command line arguments
  args = parse_args()

  if args is None:
      exit()

  # parse configuration file arguments
  config = parse_conf(root, args)

  # Setup the output directories of the program
  if config['clear']:
    clear_dirs()
  for n in ['results_path', 'results_dir', 'models_path', 'models_dir', 'tensorboard_dir', 'train_tb_dir', 'test_tb_dir']:
    d = config[n]
    make_dir(d)

  # main
  run_timit(config)
