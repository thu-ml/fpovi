from __future__ import absolute_import

import sys
import numpy as np
import tensorflow as tf
from absl import flags
import zhusuan as zs

from bandits.core.bayesian_nn import BayesianNN
from bandits.algorithms import nnet


FLAGS = flags.FLAGS



class SVGDModel(BayesianNN):
  """Implements an approximate Bayesian NN via Black-Box Alpha-Divergence."""

  def __init__(self, hparams, name):

    self.name = name
    self.hparams = hparams

    self.alpha = getattr(self.hparams, 'alpha', 1.0)
    self.num_mc_nn_samples = getattr(self.hparams, 'num_mc_nn_samples', 10)

    self.n_in = self.hparams.context_dim
    self.n_out = self.hparams.num_actions
    self.layers = self.hparams.layer_sizes
    self.batch_size = self.hparams.batch_size

    self.show_training = self.hparams.show_training
    self.freq_summary = self.hparams.freq_summary
    self.verbose = getattr(self.hparams, 'verbose', True)

    self.cleared_times_trained = self.hparams.cleared_times_trained
    self.initial_training_steps = self.hparams.initial_training_steps
    self.training_schedule = np.linspace(self.initial_training_steps,
                                         self.hparams.training_epochs,
                                         self.cleared_times_trained)

    self.times_trained = 0
    self.initialize_model()

  def initialize_model(self):
    self.graph = tf.Graph()
    with self.graph.as_default():
      from bandits.algorithms import utils
      self.sess = utils.create_session()

      self.x = tf.placeholder(shape=[None, self.n_in],
                              dtype=tf.float32, name='x')
      self.y = tf.placeholder(shape=[None, self.n_out],
                              dtype=tf.float32, name='y')
      self.weights = tf.placeholder(shape=[None, self.n_out],
                                    dtype=tf.float32, name='w')
      self.data_size = tf.placeholder(tf.float32, shape=(), name='data_size')

      self.infer_op, all_preds, self.bnn_locals = nnet.build_bnn(
        self.x, self.y, self.weights, self.data_size, self.hparams)
      self.global_step = self.bnn_locals['global_step']
      self.log_likelihood = self.bnn_locals['log_likelihood']

      # NOTE: not sure if we should sample from joint or marginals; var and
      # bootstrap in this lib used joint
      # idc = tf.random_uniform(
      #   [self.n_out], 0, self.hparams.n_particles, dtype=tf.int32)
      # self.y_pred = tf.concat(
      #   [all_preds[idc[i], :, i:i+1] for i in range(self.n_out)],
      #   axis=-1)
      idc = tf.random_uniform([], 0, self.hparams.n_particles, dtype=tf.int32)
      self.y_pred = all_preds[idc]

      self.summary_op = tf.summary.merge_all()
      self.summary_writer = tf.summary.FileWriter('{}/graph_{}'.format(
        FLAGS.logdir, self.name), self.sess.graph)

      self.sess.run(tf.global_variables_initializer())

  def assign_lr(self):
    """Resets the learning rate in dynamic schedules for subsequent trainings.

    In bandits settings, we do expand our dataset over time. Then, we need to
    re-train the network with the new data. Those algorithms that do not keep
    the step constant, can reset it at the start of each training process.
    """
    """ Not used by BBAlpha
    decay_steps = 1
    if self.hparams.activate_decay:
      current_gs = self.sess.run(self.global_step)
      with self.graph.as_default():
        self.lr = tf.train.inverse_time_decay(self.hparams.initial_lr,
                                              self.global_step - current_gs,
                                              decay_steps,
                                              self.hparams.lr_decay_rate)
    """

  def train(self, data, num_steps):
    """Trains the BNN for num_steps, using the data in 'data'.

    Args:
      data: ContextualDataset object that provides the data.
      num_steps: Number of minibatches to train the network for.
    """

    if self.times_trained < self.cleared_times_trained:
      num_steps = int(self.training_schedule[self.times_trained])
    self.times_trained += 1

    if self.verbose:
      print('Training {} for {} steps...'.format(self.name, num_steps))

    with self.graph.as_default():

      for step in range(num_steps):
        x, y, w = data.get_batch_with_weights(self.hparams.batch_size)
        fd = {self.x: x, self.y: y, self.weights: w,
              self.data_size: data.num_points()}
#       import IPython; IPython.embed(); raise 1
        _, summary, global_step, loss = self.sess.run(
          [self.infer_op, self.summary_op, self.global_step, self.log_likelihood],
          feed_dict=fd)

        if step % self.freq_summary == 0:
          if self.show_training:
            print('{} | step: {}, loss: {}'.format(self.name, global_step, loss))
            sys.stdout.flush()
          self.summary_writer.add_summary(summary, global_step)
