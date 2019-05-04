import tensorflow as tf
import numpy as np
import zhusuan as zs

from bandits.algorithms.rf import init_bnn_weight
from bandits.algorithms import svgd
from bandits.algorithms.utils import *

tfd = tf.contrib.distributions  # update to: tensorflow_probability.distributions


@zs.meta_bayesian_net(scope="bnn", reuse_variables=True)
def bnn_meta_model(x, layer_sizes, n_particles, n_supervised, w_prior_sd):
  """
  :param x: [batch_size, n_dims] (?)
  :param layer_sizes: list of length n_layers+1
  """
  assert int(x.shape[1]) == layer_sizes[0]

  bn = zs.BayesianNet()
  h = tf.tile(x[None, ...], [n_particles, 1, 1])

  w_scale = w_prior_sd * tf.ones([n_particles], dtype=tf.float32)

  for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
    sd = tf.tile(w_scale[:, None, None], [1, n_in+1, n_out])
    if i != len(layer_sizes) - 2:
      sd /= np.sqrt(n_out).astype('f')
    w = bn.normal(
      "w" + str(i), tf.zeros([n_particles, n_in+1, n_out]), std=sd,
      group_ndims=2)
    h = tf.concat([h, tf.ones(tf.shape(h)[:-1])[..., None]], -1) @ w
    if i != len(layer_sizes) - 2:
      h = tf.nn.relu(h)

  bn.deterministic("y_mean_all", h)
  y_mean = bn.deterministic("y_mean_sup", h[:, :n_supervised])
  y_std = bn.inverse_gamma(
    "y_std",
    alpha=tf.ones([1, layer_sizes[-1]]),
    beta=0.1 * tf.ones([1, layer_sizes[-1]]),
    n_samples=n_particles,
    group_ndims=1)
  y = bn.normal("y", y_mean, std=y_std)

  return bn


def build_bnn(x_ph, y_ph, weight_ph, n_train_ph, hps):

  inp, n_supervised = inplace_perturb(x_ph, hps.interp_batch_size, n_train_ph)
  layer_sizes = [x_ph.get_shape().as_list()[1]] + hps.layer_sizes + \
    [y_ph.get_shape().as_list()[1]]
  out_mask = weight_ph[None, :n_supervised, :]

  # ============== MODEL =======================
  weight_sd = np.sqrt(hps.prior_variance)
  meta_model = bnn_meta_model(
    inp, layer_sizes, hps.n_particles, n_supervised, weight_sd)

  def log_likelihood_fn(bn):
    log_py_xw = bn.cond_log_prob('y')
    assert len(log_py_xw.get_shape().as_list()) == 3 # [nPar, nBa, nOut]
    log_py_xw = tf.reduce_sum(log_py_xw * out_mask, axis=-1)
    return tf.reduce_mean(log_py_xw, 1) * n_train_ph

  meta_model.log_joint = log_likelihood_fn

  # ============== VARIATIONAL ==================
  svgd_latent = dict()
  svgd_variables = dict()

  if hps.use_sigma_exp_transform:
    sigma_transform = tfd.bijectors.Exp()
  else:
    sigma_transform = tfd.bijectors.Softplus()

  std_raw = tf.get_variable(
    'std_raw', shape=[hps.n_particles, 1, layer_sizes[-1]],
    initializer=tf.zeros_initializer())
  svgd_variables['y_std'] = std_raw
  y_std_sym = sigma_transform.forward(
    std_raw + sigma_transform.inverse(hps.noise_sigma))
  svgd_latent['y_std'] = y_std_sym

  # w
  for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
    w_init = init_bnn_weight(hps.n_particles, n_in, n_out) * weight_sd
    buf = tf.get_variable('buf_'+str(i), initializer=w_init)
    svgd_latent['w'+str(i)] = svgd_variables['w'+str(i)] = buf

  # combined
  observed_bn_ = {'y': y_ph[:n_supervised]}
  observed_bn_.update(svgd_latent)
  var_bn = meta_model.observe(**observed_bn_)
  log_likelihood = var_bn.log_joint()

  fval_all = var_bn['y_mean_all']
  log_joint_svgd = log_likelihood

  # ===== PRIOR GRADIENT =====
  param_names = [name for name in svgd_variables if name.startswith('w')]
  log_prior = var_bn.cond_log_prob(param_names)
  hpv = []
  for i in range(hps.mm_n_particles // hps.n_particles):
    temp_bn = meta_model.observe()
    hpv.append(temp_bn['y_mean_all'])
  hpv = tf.concat(hpv, axis=0)

  n_mms = hps.n_mm_sample // 2
  hp_val = tf.concat(
    [hpv[:, :n_mms], hpv[:, -n_mms:]], axis=1)
  mm_fval = tf.to_double(tf.concat(
    [fval_all[:, :n_mms], fval_all[:, -n_mms:]], axis=1))

  hp_val = merge_last_axes(hp_val, 1)
  mm_fval = merge_last_axes(mm_fval, 1)
  hpmean, hpcov = reduce_moments_ax0(hp_val)
  hpprec = matrix_inverse(hpcov, hps.mm_jitter)
  pd = tf.to_float(mvn_log_prob(mm_fval, hpprec, hpmean)) / tf.to_float(
    hps.n_mm_sample)
  log_joint_svgd += pd

  # ===== SVGD-F GRADIENT =====
  svgd_grad, _ = svgd._svgd_stationary(
    hps.n_particles, log_joint_svgd, [fval_all], svgd.rbf_kernel)[0]

  optimizer = tf.train.AdamOptimizer(learning_rate=hps.lr)
  global_step = tf.get_variable(
    'global_step', initializer=0, trainable=False)
  targ = tf.stop_gradient(svgd_grad + fval_all)
  infer_op = optimizer.minimize(
    tf.reduce_mean((targ - fval_all) ** 2), global_step=global_step)

  if getattr(hps, "infer_noise_sigma", False):
    with tf.control_dependencies([infer_op]):
      infer_op = tf.train.AdamOptimizer(learning_rate=hps.lr).minimize(
        -(log_likelihood + var_bn.cond_log_prob('y_std')),
        var_list=[std_raw])

  log_py_xw = tf.reduce_sum(var_bn.cond_log_prob("y") * out_mask, axis=-1)
  log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0))
  y_pred = tf.reduce_mean(var_bn['y_mean_sup'], axis=0)
  rmse = tf.sqrt(tf.reduce_mean(
    (y_pred - y_ph[:n_supervised]) ** 2 * weight_ph[:n_supervised]))

  logs = {
    'rmse': rmse,
    'log_likelihood': log_likelihood,
    'mean_std': tf.reduce_mean(y_std_sym),
    'std_first': y_std_sym[0,0,0],
    'std_last': y_std_sym[0,0,-1]
  }
  for k in logs:
    tf.summary.scalar(k, logs[k])

  return infer_op, var_bn['y_mean_sup'], locals()

  
def inplace_perturb(x, extra_batch_size, n_train, ptb_scale=0.1):
  if extra_batch_size == 0:
    return x, None

  x_dim = int(x.shape[1])
  ptb_scale = tf.to_float(ptb_scale / tf.sqrt(n_train * x_dim))
  to_perturb = x[-extra_batch_size:, :]
  x = x[:-extra_batch_size, :]
  x_perturbed = to_perturb + tf.random_normal(tf.shape(to_perturb), stddev=ptb_scale)
  model_inp = tf.stop_gradient(tf.concat([x, x_perturbed], axis=0))
  return model_inp, tf.shape(x)[0]


