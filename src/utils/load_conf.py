import argparse
from configparser import ConfigParser

import glob
import time
import datetime
import os, shutil

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
                      help="Configuration directory")

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
  ts = time.time()
  f = '%Y-%m-%d-%H-%M-%S'
  st = datetime.datetime.fromtimestamp(ts).strftime(f)
  fullname = '%s_%s' % (config['model_name'], st)
  config['fullname'] = fullname
  
  config['results_path'] = config_parser.get('architecture', 'results_path')
  config['results_dir'] = os.path.join(config['results_path'], config['fullname'])

  config['model_path'] = config_parser.get('architecture', 'model_path')
  config['model_dir'] = os.path.join(config['model_path'], config['fullname'])

  config['tensorboard_dir'] = os.path.join(config['model_dir'], 'tensorboard')
  config['train_tb_dir'] = os.path.join(config['tensorboard_dir'], 'train')
  config['test_tb_dir'] = os.path.join(config['tensorboard_dir'], 'test')
  
  config['prior_type'] = config_parser.get('model', 'prior_type')

  config['image_size'] = config_parser.getint('model', 'image_size')

  config['dim_z'] = config_parser.getint('model', 'dim_z')

  config['enc_type'] = config_parser.get('model', 'encoding_type')
  config['dec_type'] = config_parser.get('model', 'decoding_type')
  config['vae_elems'] = {'discriminator': discriminator}

  config['normalize'] = config_parser.getboolean('model', 'normalize')
    
  config['learn_rate'] = config_parser.getfloat('train', 'learn_rate')
    
  config['num_epochs'] = config_parser.getint('train', 'num_epochs')
    
  config['batch_size'] = config_parser.getint('train', 'batch_size')

  config['n_vae_train'] = config_parser.getint('train', 'n_vae_train')
  config['n_discr_train'] = config_parser.getint('train', 'n_discr_train')
  config['n_gen_train'] = config_parser.getint('train', 'n_gen_train')
    
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

  return check_conf(config)

"""checking arguments"""
def check_conf(config):
  if config['clear']:
    folders = ['model', 'results']
    for f in folders:
      for fi in os.listdir(f):
        fp = os.path.join(f, fi)
        try:
          if os.path.isfile(fp):
            os.unlink(fp)
          elif os.path.isdir(fp):
            shutil.rmtree(fp)
        except Exception as e:
          print(e)
          
  # results_path
  for d in ['results_path', 'results_dir']:
    try:
      os.mkdir(config[d])
    except(FileExistsError):
      pass

  # model_path
  for d in [
    'model_path', 'model_dir',
    'tensorboard_dir', 'train_tb_dir',
    'test_tb_dir'
  ]:
    try:
      os.mkdir(config[d])
    except(FileExistsError):
      pass


  # encoding types
  try:
    assert config['enc_type'] == 'fc' or config['enc_type'] == 'conv' or config['enc_type'] == 'inception' 
    if config['enc_type'] == 'fc':
      config['vae_elems']['encoder'] = fc_encoder
    elif config['enc_type'] == 'conv':
      config['vae_elems']['encoder'] = conv_encoder
    elif config['enc_type'] == 'inception':
      config['vae_elems']['encoder'] = inception_encoder
  except :
    print('vae must use either fully connected layers (enc_type = fc) or convolutional architecture (enc_type = conv, enc_type = inception)')

  try:
    assert config['dec_type'] == 'fc' or config['dec_type'] == 'conv' or config['dec_type'] == 'inception' 
    if config['dec_type'] == 'fc':
      config['vae_elems']['decoder'] = fc_decoder
    elif config['dec_type'] == 'conv':
      config['vae_elems']['decoder'] = conv_decoder
    elif config['dec_type'] == 'inception':
      config['vae_elems']['decoder'] = deception_decoder
  except :
    print('vae must use either fully connected layers (dec_type = fc) or convolutional architecture (dec_type = conv)')

  # --learn_rate
  try:
    assert config['learn_rate'] > 0
  except:
    print('learning rate must be positive')

  # --num_epochs
  try:
    assert config['num_epochs'] >= 1
  except:
    print('number of epochs must be larger than or equal to one')

  # --batch_size
  try:
    assert config['batch_size'] >= 1
  except:
    print('batch size must be larger than or equal to one')

  # --PRR
  try:
    assert config['PRR'] == True or config['PRR'] == False
  except:
    print('PRR must be boolean type')
    return None

  if config['PRR'] == True:
      # --PRR_n_img_x, --PRR_n_img_y
      try:
          assert config['PRR_n_img_x'] >= 1 and config['PRR_n_img_y'] >= 1
      except:
          print('PRR : number of images along each axis must be larger than or equal to one')

      # --PRR_resize_factor
      try:
          assert config['PRR_resize_factor'] > 0
      except:
          print('PRR : resize factor for each displayed image must be positive')

  # --PMLR
  try:
      assert config['PMLR'] == True or config['PMLR'] == False
  except:
      print('PMLR must be boolean type')
      return None

  if config['PMLR'] == True:

      # --PMLR_n_img_x, --PMLR_n_img_y
      try:
          assert config['PMLR_n_img_x'] >= 1 and config['PMLR_n_img_y'] >= 1
      except:
          print('PMLR : number of images along each axis must be larger than or equal to one')

      # --PMLR_resize_factor
      try:
          assert config['PMLR_resize_factor'] > 0
      except:
          print('PMLR : resize factor for each displayed image must be positive')

      # --PMLR_z_range
      try:
          assert config['PMLR_z_range'] > 0
      except:
          print('PMLR : range for unifomly distributed latent vector must be positive')

      # --PMLR_n_samples
      try:
          assert config['PMLR_n_samples'] > 100
      except:
          print('PMLR : Number of samples in order to get distribution of labeled data must be large enough')

  return config

