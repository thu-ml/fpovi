#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import sys

import tensorflow as tf
from six.moves import range, zip
import numpy as np
import zhusuan as zs
import experiments.utils
from experiments import wrapped_supervisor

import dataset
import svgd
from rf import init_bnn_weight
from utils import *


parser = experiments.utils.parser('bnn')
parser.add_argument('-layers', nargs='+', default=[50], type=int)
parser.add_argument('-n_particles', default=20, type=int)
parser.add_argument('-batch_size', default=1000, type=int, help='batch size of training set samples')
parser.add_argument('-extra_batch_size', default=-100, type=int,
                    help='batch size for the full-support distribution \\nu')
parser.add_argument('-ptb_type', default='kde', type=str,
                    choices=['kde', 'mixup'])
parser.add_argument('-psvi_method', default='svgd', type=str,
                    choices=['svgd', 'gfsf', 'wsgld', 'pisgld'])
parser.add_argument('-n_epoch', default=1000, type=int)
parser.add_argument('-dataset', default='concrete', type=str)
parser.add_argument('-lr', default=4e-3, type=float)
parser.add_argument('-lr_decay', action='store_true')
parser.add_argument('-fix_variance', default=-1, type=float,
                    help='if > 0, fix the variance of the likelihood model to that value')
parser.add_argument('-optimizer', default='adam', type=str)
parser.add_argument('-seed', default=1234, type=int)
parser.add_argument('-test_freq', default=50, type=int)
parser.add_argument('-n_mm_sample', default=4, type=int)
parser.add_argument('-mm_n_particles', default=40, type=int)
parser.add_argument('-mm_jitter', default=1e-3, type=float)
parser.add_argument('-valid', action='store_true', dest='use_valid')
parser.add_argument('-ptb_scale', default=0.1, type=float)
parser.add_argument('-logits_w_sd', default=40, type=float)
parser.add_argument('-dump_pred_dir', default='', type=str)
parser.add_argument('-data_dir',
                    default=os.path.join(
                        experiments.utils.source_dir(), '../data'),
                    type=str)
parser.add_argument('-save', action='store_true', dest='save_model')
parser.set_defaults(use_valid=False, lr_decay=False, regression=True, save_model=False)


@zs.meta_bayesian_net(scope="bnn", reuse_variables=True)
def build_model(x, layer_sizes, n_particles, real_batch_size, is_regression,
                logits_w_sd):
  """
  :param x: [batch_size, n_dims] (?)
  :param layer_sizes: list of length n_layers+1
  """
  assert int(x.shape[1]) == layer_sizes[0]

  bn = zs.BayesianNet()
  h = tf.tile(x[None, ...], [n_particles, 1, 1])
  for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
    std = np.sqrt(1 / n_out)
    if not is_regression and i == len(layer_sizes) - 2:
      std *= logits_w_sd

    w = bn.normal("w" + str(i),
                  tf.zeros([n_in+1, n_out]),
                  std=std.astype('f'),
                  group_ndims=2, n_samples=n_particles)
    h = tf.concat([h, tf.ones(tf.shape(h)[:-1])[..., None]], -1) @ w
    if i != len(layer_sizes) - 2:
      h = tf.nn.relu(h)

  if is_regression:
    assert h.get_shape().as_list()[-1] == 1
    h = tf.squeeze(h, -1)
    bn.deterministic("y_mean_all", h)
    y_mean = bn.deterministic("y_mean_sup", h[:, :real_batch_size])
    y_std = bn.inverse_gamma(
      "y_std", alpha=[1.], beta=[0.1], n_samples=n_particles, group_ndims=1)
    assert len(y_std.shape) == 2
    y = bn.normal("y", y_mean, std=y_std)
  else:
    h = tf.nn.log_softmax(h, axis=-1)
    bn.deterministic("y_mean_all", h)
    y_mean = bn.deterministic("y_mean_sup", h[:, :real_batch_size])
    y = bn.categorical("y", logits=y_mean)

  return bn


class Model:

  def __init__(self, hps, S):

    # Build the computation graph
    x = tf.placeholder(tf.float32, shape=[None, S.x_dim])
    if hps.regression:
      y = tf.placeholder(tf.float32, shape=[None])
      layer_sizes = [S.x_dim] + hps.layers + [1]
    else:
      y = tf.placeholder(tf.int32, shape=[None])
      layer_sizes = [S.x_dim] + hps.layers + [S.y_dim]

    # ===== MODEL & VARIATIONAL =====
    svgd_latent = dict()
    svgd_variables = dict()

    if hps.regression:
      std_raw = tf.get_variable(
        'std_raw', shape=[hps.n_particles, 1],
        initializer=tf.constant_initializer(inv_softplus(0.5)))
      svgd_variables['y_std'] = std_raw
      y_std_sym = tf.nn.softplus(std_raw)
      if hps.fix_variance > 0:
        y_std_sym = tf.clip_by_value(y_std_sym, hps.fix_variance, hps.fix_variance + 1e-5)
      svgd_latent['y_std'] = y_std_sym

    real_batch_size = tf.shape(x)[0]
    inp, x_extra = add_perturb_input(
      x, hps.extra_batch_size, S.n_train, ptb_type=hps.ptb_type,
      ptb_scale=hps.ptb_scale)
    meta_model = build_model(
      inp, layer_sizes, hps.n_particles, real_batch_size, hps.regression,
      hps.logits_w_sd)

    def log_likelihood_fn(bn):
      log_py_xw = bn.cond_log_prob('y')
      return tf.reduce_mean(log_py_xw, 1) * S.n_train

    meta_model.log_joint = log_likelihood_fn

    # variational: w
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
      buf = tf.get_variable(
        'buf_'+str(i),
        initializer=init_bnn_weight(hps.n_particles, n_in, n_out))
      svgd_latent['w'+str(i)] = svgd_variables['w'+str(i)] = buf

    # combined
    observed_bn_ = {'y': y}; observed_bn_.update(svgd_latent)
    var_bn = meta_model.observe(**observed_bn_)
    log_likelihood = var_bn.log_joint()
    fval_all = var_bn['y_mean_all']

    log_joint_svgd = log_likelihood

    # ===== PRIOR GRADIENT =====
    fv_all_prior = tf.concat(
      [meta_model.observe()['y_mean_all'] for i in range(hps.mm_n_particles // hps.n_particles)],
      axis=0)
    pg_indices = tf.concat(
      [tf.range(hps.n_mm_sample//2),
       tf.range(tf.shape(fval_all)[1] - hps.n_mm_sample//2, tf.shape(fval_all)[1])],
      axis=0)
    prior_fval = tf.gather(fv_all_prior, pg_indices, axis=1)
    var_fval = tf.to_double(tf.gather(fval_all, pg_indices, axis=1))
    if not hps.regression:
      prior_fval = merge_last_axes(prior_fval, 1)
      var_fval = merge_last_axes(var_fval, 1)
    hpmean, hpcov = reduce_moments_ax0(prior_fval)
    hpprec = matrix_inverse(hpcov, hps.mm_jitter)
    log_joint_svgd += tf.to_float(mvn_log_prob(var_fval, hpprec, hpmean))

    # ===== SVGD-F GRADIENT =====
    svgd_grad, _ = svgd._svgd_stationary(
      hps.n_particles, log_joint_svgd, [fval_all], svgd.rbf_kernel,
      additional_grad=None, method=hps.psvi_method)[0]

    # ===== INFER OP =====
    optimizer_class = {
      'adam': tf.train.AdamOptimizer,
      'adagrad': tf.train.AdagradOptimizer
    }[hps.optimizer]
    global_step = tf.get_variable(
      'global_step', initializer=0, trainable=False) 
    if hps.lr_decay:
      lr_sym = tf.train.exponential_decay(hps.lr, global_step, 10000, 0.2, staircase=True)
    else:
      lr_sym = hps.lr
    optimizer = optimizer_class(learning_rate=lr_sym)
    targ = tf.stop_gradient(svgd_grad + fval_all)
    infer_op = optimizer.minimize(
      tf.reduce_mean((targ - fval_all) ** 2), global_step=global_step)

    if hps.regression:
      # the target above doesn't include std. Use MAP
      with tf.control_dependencies([infer_op]):
        infer_op = optimizer_class(learning_rate=lr_sym).minimize(
          -(log_likelihood + var_bn.cond_log_prob('y_std')),
          var_list=[std_raw])

    # prediction: rmse & log likelihood
    log_py_xw = var_bn.cond_log_prob("y")
    log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0))
    if hps.regression:
      log_likelihood -= tf.log(S.std_y_train)
      y_pred = tf.reduce_mean(var_bn['y_mean_sup'], 0)
      rmse = tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) * S.std_y_train
      ystd_avg = tf.reduce_mean(y_std_sym)
    else:  
      y_pred = tf.reduce_mean(tf.exp(var_bn['y_mean_sup']), axis=0)
      rmse = 1 - tf.reduce_mean(tf.to_float(tf.nn.in_top_k(y_pred, y, 1)))
      ystd_avg = tf.constant(-1.)

    self.__dict__.update(locals())


def load_data(hps):

  data_path = os.path.join(hps.data_dir, hps.dataset + '.data')
  data_func = dataset.data_dict()[hps.dataset]
  x_train, y_train, x_valid, y_valid, x_test, y_test = data_func(data_path)

  if not hps.use_valid:
    x_train = np.vstack([x_train, x_valid])
    y_train = np.concatenate([y_train, y_valid], axis=0)
    x_valid = y_valid = None

  n_train, x_dim = x_train.shape
  if y_test.dtype in [np.int32, np.int64]:
    hps.regression = False
    y_dim = y_train.max() + 1
  else:
    y_dim = 1

  if hps.dataset == 'mnist':
    # loaded data has range [0, 1]. previous work does not normalize individual
    # pixels
    xm = x_train.mean()
    x_train -= xm
    x_test -= xm
    if x_valid is not None:
      x_valid -= xm
    S = Object()
  else:
    x_train, y_train, x_valid, y_valid, x_test, y_test, S = \
      dataset.standardize_new(
        x_train, y_train, x_valid, y_valid, x_test, y_test, hps.regression)

  S.__dict__.update({
    'n_train': n_train,
    'x_dim': x_dim,
    'y_dim': y_dim})

  return x_train, y_train, x_valid, y_valid, x_test, y_test, S


def main(hps):
  tf.set_random_seed(hps.seed)
  np.random.seed(hps.seed)
  x_train, y_train, x_valid, y_valid, x_test, y_test, S = load_data(hps)
  M = Model(hps, S)

  # Define training/evaluation parameters
  iters = int(np.ceil(x_train.shape[0] / float(hps.batch_size)))

  # Run the inference
  sms = 60 if hps.save_model else 0
  with wrapped_supervisor.create_sv(
      hps, save_model_secs=sms, save_summaries_secs=0,
      global_step=M.global_step) as sv:
    sess = sv.sess_
    train_loop(
      sess=sess, hps=hps,
      x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
      infer_op=M.infer_op, rmse_op=M.rmse, loglh_op=M.log_likelihood,
      ystd_op=M.ystd_avg, x_ph=M.x, y_ph=M.y, x_extra_ph=M.x_extra,
      x_valid=x_valid, y_valid=y_valid)

    if len(hps.dump_pred_dir) > 0:
      if not hps.regression or hps.dataset == 'mnist':
        raise NotImplementedError()

      pred_out = sv.sess_.run([M.var_bn['y_mean_sup'], M.ystd_avg],
                              {M.x: x_test, M.y: y_test})
      pred_out[0] = pred_out[0] * S.std_y_train + S.mean_y_train
      pred_out[1] *= S.std_y_train
      f = lambda a, b: [a*S.std_x_train + S.mean_x_train,
                        b*S.std_y_train + S.mean_y_train]
      todump = pred_out + f(x_test, y_test) + f(x_train, y_train)

      with open(hps.dump_pred_dir, 'wb') as fout:
        pickle.dump(todump, fout)


if __name__ == "__main__":
  hps = parser.parse_args()
  experiments.utils.preflight(hps)
  main(hps)
