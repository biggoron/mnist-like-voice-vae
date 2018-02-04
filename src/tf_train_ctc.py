#!/usr/bin/env python

import os
import sys
import numpy as np
import time
import warnings
from configparser import ConfigParser
import logging

import tensorflow as tf
from tensorflow.python.ops import ctc_ops

# Custom modules
from utils.text import ndarray_to_text, sparse_tuple_to_texts, ndarray_to_phn, sparse_tuple_to_phn

# in future different than utils class
from utils.utils import create_optimizer
#from data.datasets import read_datasets
from utils.set_dirs import get_conf_dir, get_model_dir
import utils.gpu as gpu_tool

import data.dataset2 as dt2
# Import the setup scripts for different types of model
from networks.rnn2 import BiRNN as BiRNN_model
from networks.rnn2 import SimpleLSTM as SimpleLSTM_model

logger = logging.getLogger(__name__)


class Tf_train_ctc(object):
    '''
    Class to train a speech recognition model with TensorFlow.

    Requirements:
    - TensorFlow 1.0.1
    - Python 3.5
    - Configuration: $RNN_TUTORIAL/configs/neural_network.ini

    Features:
    - Batch loading of input data
    - Checkpoints model
    - Label error rate is the edit (Levenshtein) distance of the top path vs true sentence
    - Logs summary stats for TensorBoard
    - Epoch 1: Train starting with shortest transcriptions, then shuffle

    # Note: All calls to tf.name_scope or tf.summary.* support TensorBoard visualization.

    This class was initially based on open source code from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/DeepSpeech.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    '''

    def __init__(self,
                 config_file='neural_network.ini',
                 model_name=None,
                 debug=False):
        # set TF logging verbosity
        tf.logging.set_verbosity(tf.logging.INFO)

        # Load the configuration file depending on debug True/False
        self.debug     = debug
        self.conf_dir = get_conf_dir(debug = self.debug)
        self.conf_path = os.path.join(self.conf_dir, config_file)
        self.load_configs()
        self.regularizable = [
            "h1/read:0",
            "h2/read:0",
            "h3/read:0",
            "h5/read:0",
            "h6/read:0",
            "h7/read:0",
            "lstm1/fw/basic_lstm_cell/weights/read:0",
            "lstm2/fw/basic_lstm_cell/weights/read:0",
            "lstm3/fw/basic_lstm_cell/weights/read:0",
            "lstm1/bw/basic_lstm_cell/weights/read:0",
            "lstm2/bw/basic_lstm_cell/weights/read:0",
            "lstm3/bw/basic_lstm_cell/weights/read:0"
        ]

        # Verify that the GPU is operational, if not use CPU
        if not gpu_tool.check_if_gpu_available(self.tf_device):
            self.tf_device = '/cpu:0'
        logging.info('Using this device for main computations: %s', self.tf_device)

        # set the directories
        self.set_up_directories(model_name)

        # set up the model
        self.set_up_model()

    def load_configs(self):
        parser = ConfigParser(os.environ)
        if not os.path.exists(self.conf_path):
            raise IOError("Configuration file '%s' does not exist" % self.conf_path)
        logging.info('Loading config from %s', self.conf_path)
        parser.read(self.conf_path)

        # set which set of configs to import
        h = 'nn'
        self.network_type = parser.get(h, 'network_type')

        h = 'training'
        self.epochs              = parser.getint(h, 'epochs')
        self.batch_size          = parser.getint(h, 'batch_size')
        self.n_batches_per_epoch = parser.getint(h, 'n_batches_per_epoch')
        # How often to save the model
        self.save_model_epoch_num = parser.getint( h, 'save_model_epoch_num')
        # decode dev set after N epochs
        self.validation_epoch_num = parser.getint( h, 'validation_epoch_num')
        # decide when to stop training prematurely
        self.curr_val_ler_diff = parser.getfloat(h, 'curr_val_ler_diff')
        self.avg_val_ler_epochs = parser.getint(h, 'avg_val_ler_epochs')
        # initialize list to hold average validation at end of each epoch
        self.avg_val_lers = [ 1.0 for _ in range(self.avg_val_ler_epochs)]
        # determine if the data input order should be shuffled after every epic
        self.shuffle_data = parser.getboolean(h, 'shuffle_data')
        # initialize to store the minimum validation set label error rate
        self.min_dev_ler = parser.getfloat(h, 'min_dev_ler')

        h = 'data'
        # Number of features, 13 or 26
        self.n_input   = parser.getint(h, 'n_input')
        self.emb = parser.getboolean(h, 'embedding')
        # Number of contextual samples to include
        self.n_context = parser.getint(h, 'n_context')
        self.txt_phn = parser.getboolean(h, 'phoneme')

        # self.decode_train = parser.getboolean(config_header, 'decode_train')
        # self.random_seed = parser.getint(config_header, 'random_seed')
        h = 'dirs'
        self.model_dir = parser.get(h, 'model_dir')

        # set the session name
        self.session_name = '{}_{}'.format(
            self.network_type, time.strftime("%Y%m%d-%H%M%S"))
        sess_prefix_str = 'develop'
        if len(sess_prefix_str) > 0:
            self.session_name = '{}_{}'.format(
                sess_prefix_str, self.session_name)

        h = 'reg'
        # L2 regularization
        self.l2_reg = parser.getfloat( h, 'lambda_l2_reg')

        h = 'beam_search'
        # setup type of decoder
        self.beam_search_decoder = parser.get(h, 'beam_search_decoder')

        h = 'gpu'
        # set up GPU if available
        self.tf_device = str(parser.get(h, 'tf_device'))
        # set up the max amount of simultaneous users
        # this restricts GPU usage to the inverse of self.simultaneous_users_count
        self.users_count = parser.getint(h, 'users_count')

    def set_up_directories(self, model_name):
        # Set up model directory
        self.model_dir = os.path.join(get_model_dir(), self.model_dir)
        # summary will contain logs
        self.summary_dir = os.path.join(
            self.model_dir, "summary", self.session_name)
        # session will contain models
        self.session_dir = os.path.join(
            self.model_dir, "session", self.session_name)

        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        # set the model name and restore if not None
        if model_name is not None:
            self.model_path = os.path.join(self.model_dir, "session", model_name, )
        else:
            self.model_path = None

    def set_up_model(self):
        self.sets = ['timit_data', 'test-clean']

        self.datastore = dt2.DataStore()
        self.datastore.create_collection('train')
        self.datastore.create_collection('test')
        self.datastore.dir_to_collection('data/timit_data/', 'train', emb = self.emb)
        self.datastore.dir_to_collection('data/test-clean/', 'test', emb = self.emb)

        self.datastore.shuffle_collection('train')

        self.train_dataset = self.datastore.dataset_rnn('train', step = 1, ftnb = self.n_input, width = self.n_context, txt_phn = self.txt_phn)
        self.test_dataset = self.datastore.dataset_rnn('test', step = 1, ftnb = self.n_input, width = self.n_context, txt_phn = self.txt_phn)
        # read data set, inherits configuration path
        # to parse the config file for where data lives
#        self.data_sets = read_datasets(self.conf_path,
#                                       self.sets,
#                                       self.n_input,
#                                       self.n_context,
#                                       )

        self.n_examples_train = self.train_dataset.l
        self.n_examples_test = self.test_dataset.l

        logger.info('''Training model: {}
        Train samples: {:,}
        Test samples: {:,}
        Epochs: {}
        Training batch size: {}
        Batches per epoch: {}'''.format(
            self.session_name,
            self.n_examples_train,
            self.n_examples_test,
            self.epochs,
            self.batch_size,
            self.n_batches_per_epoch))

    def run_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):
            with tf.device(self.tf_device):
                # Run multiple functions on the specificed tf_device
                # tf_device GPU set in configs, but is overridden if not available
                self.setup_network_and_graph()
                self.load_placeholder_into_network()
                self.setup_loss_function()
                self.setup_optimizer()
                self.setup_decoder()

            self.setup_summary_statistics()

            # create the configuration for the session
            tf_config = tf.ConfigProto()
            tf_config.allow_soft_placement = True
            tf_config.gpu_options.per_process_gpu_memory_fraction = \
                (1.0 / self.users_count)

            # create the session
            self.sess = tf.Session(config=tf_config)

            # initialize the summary writer
            self.writer = tf.summary.FileWriter(
                self.summary_dir, graph=self.sess.graph)

            # Add ops to save and restore all the variables
            self.saver = tf.train.Saver()

            # For printing out section headers
            section = '\n{0:=^40}\n'

            # If there is a model_path declared, then restore the model
            if self.model_path is not None:
                print('YOSH!!!')
                print('YOSH!!!')
                print('YOSH!!!')
                print('YOSH!!!')
                print('YOSH!!!')
                print('YOSH!!!')
                self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))
            # If there is NOT a model_path declared, build the model from scratch
            else:
                # Op to initialize the variables
                init_op = tf.global_variables_initializer()

                # Initializate the weights and biases
                self.sess.run(init_op)

                # MAIN LOGIC for running the training epochs
                logger.info(section.format('Run training epoch'))
                self.run_training_epochs()

            logger.info(section.format('Decoding test data'))
            # make the assumption for working on the test data, that the epoch here is the last epoch
            _, self.test_ler = self.run_batches(
              self.test_dataset,
              is_training=False,
              decode=True,
              write_to_file=False,
              epoch=self.epochs
            )

            # Add the final test data to the summary writer
            # (single point on the graph for end of training run)
            summary_line = self.sess.run(
                self.test_ler_op, {self.ler_placeholder: self.test_ler})
            self.writer.add_summary(summary_line, self.epochs)

            logger.info('Test Label Error Rate: {}'.format(self.test_ler))

            # save train summaries to disk
            self.writer.flush()

            self.sess.close()

    def setup_network_and_graph(self):
        # e.g: log filter bank or MFCC features
        # shape = [batch_size, max_stepsize, n_input + (2 * n_input * n_context)]
        # the batch_size and max_stepsize can vary along each step
        self.input_tensor = tf.placeholder(
            tf.float32, [None, None, self.n_input + (2 * self.n_input * self.n_context)], name='input')

        # Use sparse_placeholder; will generate a SparseTensor, required by ctc_loss op.
        self.targets = tf.sparse_placeholder(tf.int32, name='targets')
        # 1d array of size [batch_size]
        self.seq_length = tf.placeholder(tf.int32, [None], name='seq_length')

    def load_placeholder_into_network(self):
        # logits is the non-normalized output/activations from the last layer.
        # logits will be input for the loss function.
        # nn_model is from the import statement in the load_model function
        # summary_op variables are for tensorboard
        if self.network_type == 'SimpleLSTM':
            self.logits, summary_op = SimpleLSTM_model(
                self.conf_path,
                self.input_tensor,
                tf.to_int64(self.seq_length)
            )
        elif self.network_type == 'BiRNN':
            self.logits, summary_op = BiRNN_model(
                self.conf_path,
                self.input_tensor,
                tf.to_int64(self.seq_length),
                self.n_input,
                self.n_context
            )
        else:
            raise ValueError('network_type must be SimpleLSTM or BiRNN')
        self.summary_op = tf.summary.merge([summary_op])

    def setup_loss_function(self):
        with tf.name_scope("loss"):
            self.total_loss = ctc_ops.ctc_loss(
                self.targets, self.logits, self.seq_length)
            for variable in tf.trainable_variables():
                logging.info('Candidate variable: %s', variable)
            l2 = self.l2_reg * sum(
                tf.nn.l2_loss(tf_var)
                for tf_var in tf.trainable_variables()
                if tf_var.name in self.regularizable
            )
            self.avg_loss = tf.reduce_mean(self.total_loss) + l2
            self.loss_summary = tf.summary.scalar("avg_loss", self.avg_loss)

            self.cost_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

            self.train_cost_op = tf.summary.scalar(
                "train_avg_loss", self.cost_placeholder)

    def setup_optimizer(self):
        # Note: The optimizer is created in models/RNN/utils.py
        with tf.name_scope("train"):
            self.optimizer = create_optimizer()
            self.optimizer = self.optimizer.minimize(self.avg_loss)

    def setup_decoder(self):
        with tf.name_scope("decode"):
            if self.beam_search_decoder == 'default':
                self.decoded, self.log_prob = ctc_ops.ctc_beam_search_decoder(
                    self.logits, self.seq_length, merge_repeated=False)
            elif self.beam_search_decoder == 'greedy':
                self.decoded, self.log_prob = ctc_ops.ctc_greedy_decoder(
                    self.logits, self.seq_length, merge_repeated=False)
            else:
                logging.warning("Invalid beam search decoder option selected!")

    def setup_summary_statistics(self):
        # Create a placholder for the summary statistics
        with tf.name_scope("accuracy"):
            # Compute the edit (Levenshtein) distance of the top path
            distance = tf.edit_distance(
                tf.cast(self.decoded[0], tf.int32), self.targets)

            # Compute the label error rate (accuracy)
            self.ler = tf.reduce_mean(distance, name='label_error_rate')
            self.ler_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
            self.train_ler_op = tf.summary.scalar(
                "train_label_error_rate", self.ler_placeholder)
            self.dev_ler_op = tf.summary.scalar(
                "validation_label_error_rate", self.ler_placeholder)
            self.test_ler_op = tf.summary.scalar(
                "test_label_error_rate", self.ler_placeholder)

    def run_training_epochs(self):
        train_start = time.time()
        for epoch in range(self.epochs):
            # Initialize variables that can be updated
            save_dev_model = False
            stop_training = False
            is_checkpoint_step, is_validation_step = \
                self.validation_and_checkpoint_check(epoch)

            epoch_start = time.time()

            self.train_dataset.reset_index()
            self.train_cost, self.train_ler = self.run_batches(
                self.train_dataset,
                is_training=True,
                decode=False,
                write_to_file=False,
                epoch=epoch)

            epoch_duration = time.time() - epoch_start

            log = 'Epoch {}/{}, train_cost: {:.3f}, \
                   train_ler: {:.3f}, time: {:.2f} sec'
            logger.info(log.format(
                epoch + 1,
                self.epochs,
                self.train_cost,
                self.train_ler,
                epoch_duration))

            summary_line = self.sess.run(
                self.train_ler_op, {self.ler_placeholder: self.train_ler})
            self.writer.add_summary(summary_line, epoch)

            summary_line = self.sess.run(
                self.train_cost_op, {self.cost_placeholder: self.train_cost})
            self.writer.add_summary(summary_line, epoch)

            # Shuffle the data for the next epoch
            if self.shuffle_data:
                self.train_dataset.shuffle_data()

            # Run validation if it was determined to run a validation step
            if is_validation_step:
                self.run_validation_step(epoch)

            if (epoch + 1) == self.epochs or is_checkpoint_step:
                # save the final model
                save_path = self.saver.save(self.sess, os.path.join(
                    self.session_dir, 'model.ckpt'), epoch)
                logger.info("Model saved: {}".format(save_path))

            if save_dev_model:
                # If the dev set is not improving,
                # the training is killed to prevent overfitting
                # And then save the best validation performance model
                save_path = self.saver.save(self.sess, os.path.join(
                    self.session_dir, 'model-best.ckpt'))
                logger.info(
                    "Model with best validation label error rate saved: {}".
                    format(save_path))

            if stop_training:
                break

        train_duration = time.time() - train_start
        logger.info('Training complete, total duration: {:.2f} min'.format(
            train_duration / 60))

    def run_validation_step(self, epoch):
        dev_ler = 0

        self.test_dataset.reset_index()
        _, dev_ler = self.run_batches(self.test_dataset,
                                      is_training=False,
                                      decode=True,
                                      write_to_file=False,
                                      epoch=epoch)

        logger.info('Validation Label Error Rate: {}'.format(dev_ler))

        summary_line = self.sess.run(
            self.dev_ler_op, {self.ler_placeholder: dev_ler})
        self.writer.add_summary(summary_line, epoch)

        if dev_ler < self.min_dev_ler:
            self.min_dev_ler = dev_ler

        # average historical LER
        history_avg_ler = np.mean(self.avg_val_lers)

        # if this LER is not better than average of previous epochs, exit
        if history_avg_ler - dev_ler <= self.curr_val_ler_diff:
            log = "Validation label error rate not improved by more than {:.2%} \
                  after {} epochs. Exit"
            warnings.warn(log.format(self.curr_val_ler_diff,
                                     self.avg_val_ler_epochs))
            sys.exit(0)

        # save avg validation accuracy in the next slot
        self.avg_val_lers[
            epoch % self.avg_val_ler_epochs] = dev_ler

    def validation_and_checkpoint_check(self, epoch):
        # initially set at False unless indicated to change
        is_checkpoint_step = False
        is_validation_step = False

        # Check if the current epoch is a validation or checkpoint step
        if (epoch > 0) and ((epoch + 1) != self.epochs):
            if (epoch + 1) % self.save_model_epoch_num == 0:
                is_checkpoint_step = True
            if (epoch + 1) % self.validation_epoch_num == 0:
                is_validation_step = True

        return is_checkpoint_step, is_validation_step


    def run_batches(self, dataset, is_training, decode, write_to_file, epoch):
        n_examples = len(dataset._txt_files)

        n_batches_per_epoch = int(np.ceil(n_examples / self.batch_size))

        self.train_cost = 0
        self.train_ler = 0
        
        
        for batch in range(n_batches_per_epoch):
            # Get next batch of training data (audio features) and transcripts
            logger.debug('Batch %i / %i' % (batch, n_batches_per_epoch))
            logger.debug('New Batch')
            source, source_lengths, sparse_labels = dataset.next_batch(self.batch_size)
            logger.debug('New Batch Prepared')
            feed = {self.input_tensor: source,
                    self.targets: sparse_labels,
                    self.seq_length: source_lengths}

            # If the is_training is false, this means straight decoding without computing loss
            if is_training:
                # avg_loss is the loss_op, optimizer is the train_op;
                # running these pushes tensors (data) through graph
                ler, batch_cost, _ = self.sess.run(
                    [self.ler, self.avg_loss, self.optimizer], feed)
                self.train_cost += batch_cost * self.batch_size
                logger.debug('Batch cost: %.2f | Train cost: %.2f',
                             batch_cost, self.train_cost)
            else:
                ler = self.sess.run(self.ler, feed_dict = feed)

            self.train_ler += ler * self.batch_size
            logger.debug('Label error rate: %.2f', self.train_ler)

            # Turn on decode only 1 batch per epoch
            if decode and batch == 0:
                d = self.sess.run(self.decoded[0], feed_dict={
                    self.input_tensor: source,
                    self.targets: sparse_labels,
                    self.seq_length: source_lengths}
                )
                dense_decoded = tf.sparse_tensor_to_dense(
                    d, default_value=-1).eval(session=self.sess)
                if self.txt_phn:
                  dense_labels = sparse_tuple_to_phn(sparse_labels)
                else:
                  dense_labels = sparse_tuple_to_texts(sparse_labels)

                # only print a set number of example translations
                counter = 0
                counter_max = 4
                if counter < counter_max:
                    for orig, decoded_arr in zip(dense_labels, dense_decoded):
                        # convert to strings
                        if self.txt_phn:
                          decoded_str = ndarray_to_phn(decoded_arr)
                        else:
                          decoded_str = ndarray_to_text(decoded_arr)
                        logger.info('Batch {}, file {}'.format(batch, counter))
                        logger.info('Original: {}'.format(orig))
                        logger.info('Decoded:  {}'.format(decoded_str))
                        counter += 1

                # save out variables for testing
                self.dense_decoded = dense_decoded
                self.dense_labels = dense_labels

        # Metrics mean
        if is_training:
            self.train_cost /= n_examples
        self.train_ler /= n_examples

        # Populate summary for histograms and distributions in tensorboard
        self.accuracy, summary_line = self.sess.run(
            [self.avg_loss, self.summary_op], feed)
        self.writer.add_summary(summary_line, epoch)

        return self.train_cost, self.train_ler


# to run in console
if __name__ == '__main__':
    import click

    # Use click to parse command line arguments
    @click.command()
    @click.option('--config', default='neural_network.ini', help='Configuration file name')
    @click.option('--name', default=None, help='Model name for logging')
    @click.option('--debug', type=bool, default=False,
                  help='Use debug settings in config file')
    # Train RNN model using a given configuration file
    def main(config='neural_network.ini', name=None, debug=False):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        global logger
        logger = logging.getLogger(os.path.basename(__file__))

        # create the Tf_train_ctc class
        tf_train_ctc = Tf_train_ctc(
            config_file=config, model_name=name, debug=debug)

        # run the training
        tf_train_ctc.run_model()

    main()

