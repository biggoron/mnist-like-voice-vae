import time, datetime, os, shutil
import argparse
from configparser import ConfigParser

from src.networks.vae_elements import fc_encoder, fc_decoder, conv_encoder, conv_decoder, inception_encoder, deception_decoder
from src.networks.discriminator import discriminator

"""parsing and configuration"""
def parse_args():
  desc = "Tensorflow implementation of 'Variational AutoEncoder (VAE)'"
  parser = argparse.ArgumentParser(description=desc)

  parser.add_argument('--config', type=str, default='aae.ini',
                      help="Configuration file's name")

  parser.add_argument('--configdir', type=str, default='config',
                      help="Configuration directory")

  parser.add_argument('--name', type=str, default='default',
                      help="Name the the trained model, used to name directories")

  parser.add_argument('--clear', type=bool, default=False,
                      help="clears model and result dir")

  return parser.parse_args()

def parse_conf(root, args):
  config_parser = ConfigParser()
  config = {}
  
  conf_file = args.config
  conf_dir = args.configdir
  conf_path = os.path.join(root, args.configdir, args.config)
  
  config_parser.read(conf_path)
  config['root'] = root
  config['conf_path'] = conf_path
  
  config['clear'] = args.clear

  config['model_name'] = args.name
  config['fullname'] = model_fullname(config['model_name'])
  
  config['results_path'] = config_parser.get('architecture', 'results_path')
  config['results_dir'] = os.path.join(config['results_path'], config['fullname'])

  config['models_path'] = config_parser.get('architecture', 'models_path')
  config['models_dir'] = os.path.join(config['models_path'], config['fullname'])

  config['tensorboard_dir'] = os.path.join(config['models_dir'], 'tensorboard')
  config['train_tb_dir'] = os.path.join(config['tensorboard_dir'], 'train')
  config['test_tb_dir'] = os.path.join(config['tensorboard_dir'], 'test')
  
  config['prior_type'] = config_parser.get('model', 'prior_type')

  config['image_width'] = config_parser.getint('model', 'image_width')
  config['ft_nb'] = config_parser.getint('model', 'ft_nb')

  config['dim_z'] = config_parser.getint('model', 'dim_z')

  config['enc_type'] = config_parser.get('model', 'encoding_type')
  config['dec_type'] = config_parser.get('model', 'decoding_type')
  config['vae_elems'] = {'discriminator': discriminator}

  config['learn_rate'] = config_parser.getfloat('train', 'learn_rate')
    
  config['num_epochs'] = config_parser.getint('train', 'num_epochs')
    
  config['batch_size'] = config_parser.getint('train', 'batch_size')

  config['PRR'] = config_parser.getboolean('plot', 'PRR')
    
  config['PRR_n_img_x'] = config_parser.getint('plot', 'PRR_n_img_x')
    
  config['PRR_n_img_y'] = config_parser.getint('plot', 'PRR_n_img_y')
    
  config['PRR_resize_factor'] = config_parser.getfloat('plot', 'PRR_resize_factor')
    
  config['PMLR'] = config_parser.getboolean('plot', 'PMLR')
    
  config['PMLR_n_img_x'] = config_parser.getint('plot', 'PMLR_n_img_x')
    
  config['PMLR_n_img_y'] = config_parser.getint('plot', 'PMLR_n_img_y')
    
  config['PMLR_resize_factor'] = config_parser.getfloat('plot', 'PMLR_resize_factor')
    
  config['PMLR_z_range'] = config_parser.getfloat('plot', 'PMLR_z_range')
    
  config['PMLR_n_samples'] = config_parser.getint('plot', 'PMLR_n_samples')

  config['train_dir'] =  config_parser.get('data', 'train_dir')
  config['dev_dir'] =  config_parser.get('data', 'dev_dir')
  config['test_dir'] =  config_parser.get('data', 'test_dir')

  config['train_nb'] =  config_parser.getint('data', 'train_nb')
  config['test_nb'] =  config_parser.getint('data', 'test_nb')

  return config

def model_fullname(name):
  ts = time.time()
  f = '%Y-%m-%d-%H-%M-%S'
  st = datetime.datetime.fromtimestamp(ts).strftime(f)
  fullname = '%s_%s' % (name, st)
  return fullname
