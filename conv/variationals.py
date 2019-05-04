import numpy as np
import tensorflow as tf
import zhusuan as zs
from tensorpack.utils import logger
from tensorpack.tfutils.collection import backup_collection, restore_collection
from tensorpack.tfutils.summary import add_moving_summary

import svgd
import ctx
from utils import *


default_initializer_mode = 'FAN_IN'


def default_initializer(tag, n_particles):
  # variance_scaling_initializer counts all dims when computing fan_in / n,
  # which will be used to divide factors. So we multiply factors by n_particles
  if tag == 'dense/W':
    return tf.variance_scaling_initializer(2.0 * n_particles)
  elif tag == 'dense/b':
    return tf.zeros_initializer()
  elif tag == 'conv2d/kernel':
    return tf.variance_scaling_initializer(
      2.0 * n_particles, mode=default_initializer_mode)
  elif tag == 'conv2d/b':
    return tf.zeros_initializer()


def variational_model(fn):

  def wrapped(*args, **kw):
    bnn_mm_builder, true_inp = args[:2]
    hps = kw['hps']

    # get layer specs
    # This is a mock run, and should not have side effect on the main graph
    # (e.g. accidentally creating BatchNorm EMAs that are updated from prior)
    with tf.Graph().as_default(), tf.variable_scope('dummy', reuse=False):
      fake_inp = tf.placeholder(
        shape=true_inp.get_shape().as_list(), dtype=true_inp.dtype)
      bnn_meta_model = bnn_mm_builder(fake_inp, hps.n_particles_per_dev, hps)
      bn, layer_names = bnn_meta_model.observe()
    layer_specs = [
      (bn[name].get_shape(), name, tag) for (name, tag) in layer_names]

    # get default meta model and call fn
    mm = bnn_mm_builder(true_inp, hps.n_particles_per_dev, hps)
    args = [layer_specs, mm] + list(args[2:])
    ctx.set_shape_logger(False)
    kw['true_inp'] = true_inp
    kw['bnn_mm_builder'] = bnn_mm_builder
    return fn(*args, **kw)

  return wrapped


@variational_model
def empirical(layer_spec, default_bnn_mm, observed, obs_weight, devices, **kw):
  elbos, logits = [], []
  inf_vars = []

  for dev in devices:
    scope_name = dev.name[1:].replace(':', '_')
    logger.info("BUILDING GRAPH ON " + scope_name)
    with tf.device(dev.name), tf.variable_scope(
        scope_name, reuse=tf.get_variable_scope().reuse):
      latent_dict = {}
      for (lshape, name, tag) in layer_spec:
        n_particles = int(lshape[0])
        w_buf = tf.get_variable(
          name, lshape, tf.float32,
          initializer=default_initializer(tag, n_particles))
        latent_dict[name] = w_buf
      observed_dev = observed.copy()
      observed_dev.update(latent_dict)
      var_bn, _ = default_bnn_mm.observe(**observed_dev)
      log_evid = var_bn.local_log_prob('y')
      log_prior = var_bn.local_log_prob([l for _, l, _ in layer_spec])
      elbos.append(
        tf.reduce_sum(log_evid, axis=1) * obs_weight + tf.add_n(log_prior))
      logits.append(var_bn['logits'])
    inf_vars += list(latent_dict.values())

  func_all = tf.concat(logits, axis=0)
  inf_loss = -tf.add_n(elbos)
  map_loss = inf_loss
  map_vars = ctx.get_map_variables()

  return func_all, inf_loss, inf_vars, map_loss, map_vars


@variational_model
def f_svgd(layer_specs, default_bnn_mm, observed, obs_weight, devices,
           bnn_mm_builder=None, hps=None, true_inp=None):

  funcs = []
  _log_evid_grads = []
  inf_vars = []
  replace_log_evid = 0

  # ================= Log Evidence ================
  if hps.prior_type != 'fwdgrad':
    dev_prior = devices[-hps.mm_n_prior_towers:]
    devices = devices[:-hps.mm_n_prior_towers]

  particle_towers = []
  for dev in devices:
    scope_name = dev.name[1:].replace(':', '_')
    logger.info("BUILDING EVIDENCE GRAPH ON " + scope_name)
    with tf.device(dev.name), tf.variable_scope(
        scope_name, reuse=tf.get_variable_scope().reuse):
      # particle storage
      latent_dict = {}
      for (lshape, name, tag) in layer_specs:
        w_buf = tf.get_variable(
          name, lshape, tf.float32,
          initializer=default_initializer(tag, hps.n_particles_per_dev))
        latent_dict[name] = w_buf

      observed_dev = observed.copy()
      observed_dev.update(latent_dict)
      var_bn, _ = default_bnn_mm.observe(**observed_dev)
      func_i = var_bn['logits']
      y_supervised = var_bn.local_log_prob('y')
      if hps.extra_batch_size > 0:
        y_supervised = y_supervised[:, :-hps.extra_batch_size]
      log_evid = tf.reduce_sum(y_supervised, axis=1) * obs_weight
      replace_log_evid = replace_log_evid + log_evid
      _log_evid_grads.append(tf.gradients(log_evid, func_i)[0])

      funcs.append(func_i)
      particle_towers.append((dev.name, scope_name, var_bn, latent_dict))

    inf_vars += list(latent_dict.values())

  replace_log_evid_grad = [tf.concat(_log_evid_grads, axis=0, name='rleg')]
  all_func = tf.concat(funcs, axis=0, name='all_func')

  # ============== Prior =================
  real_batch_size = tf.shape(all_func)[1]
  if hps.prior_type == 'fwdgrad':
    raise NotImplementedError() # save some trouble
  elif hps.prior_type == 'mm':
    # sample from prior
    n_particles_prior_tower = hps.n_particles_per_dev * 4
    prior_mm = bnn_mm_builder(
      true_inp[:hps.batch_size // 4], n_particles_prior_tower, hps)
    f_priors = []
    for dev in dev_prior:
      scope_name = dev.name[1:].replace(':', '_')
      logger.info("Building prior graph on {}, n_particles = {}".format(
        scope_name, n_particles_prior_tower))
      with tf.device(dev.name), tf.variable_scope(
          scope_name, reuse=tf.get_variable_scope().reuse):
        var_bn, _ = prior_mm.observe()
        f_priors.append(var_bn['logits'])
    # downsample output; data-point level:
    f_priors = tf.concat(f_priors, axis=0)[:, :hps.mm_n_inp, :]
    f_observed = all_func[:, :hps.mm_n_inp, :]
    # class level:
    if hps.mm_n_downsample_class > 0:
      true_labels = observed['y'][:hps.mm_n_inp]
      lde = label_downsampler(
        hps.classnum, true_labels, hps.mm_n_downsample_class)
      f_priors = lde(f_priors)
      f_observed = lde(f_observed)
    # squash dims
    f_priors = merge_last_axes(f_priors, 1, name='f_priors')
    f_observed = tf.to_double(merge_last_axes(f_observed, 1), name='f_obs')
    # mm objective
    mean, cov = reduce_moments_ax0(f_priors)
    prec = matrix_inverse(cov, hps.mm_jitter)
    log_prior = mvn_log_prob(f_observed, prec, mean) / tf.to_double(
        tf.shape(f_observed)[1])
    grad_fx = tf.gradients(log_prior, all_func)
    grad_fx = [tf.to_float(grad_fx[0], name='f_grad')]

  # Monitor prior grad scale
  avg_evid_grad = tf.reduce_mean(replace_log_evid_grad[0] ** 2)
  avg_prior_grad = tf.reduce_mean(grad_fx[0] ** 2)
  add_moving_summary(tf.identity(
    avg_evid_grad / (1e-9 + avg_prior_grad), name='g_evid_prior_ratio'))

  n_particles = hps.n_particles_per_dev * len(devices)
  def replaced_kernel(thx, thy):
    """
    mask the true class. shouldn't invalidate the procedure, as logits are
    normalized. NOTE: not sure if it actually helps
    """
    n_particles = int(thx.get_shape().as_list()[0])
    y = tf.tile(tf.expand_dims(observed['y'], 0), [n_particles, 1])
    msk = tf.to_float(1 - tf.one_hot(y, depth=hps.classnum))
    def mp(th):
      return svgd._squeeze(
        [msk * tf.reshape(th, [n_particles, -1, hps.classnum])],
        n_particles)
    nthx = mp(thx)
    K, _ = svgd.rbf_kernel(nthx, mp(thy))
    dyKxy = -tf.gradients(K, nthx)[0]
    return K, (lambda *args: dyKxy)

  gradf, _ = svgd._svgd_stationary(
    n_particles, None, [all_func], replaced_kernel,
    replace_grad=replace_log_evid_grad, additional_grad=grad_fx,
    method=hps.stein_method)[0]
  targ = tf.stop_gradient(all_func + gradf)

  inf_loss = tf.reduce_sum((targ - all_func) ** 2)
  map_loss = -replace_log_evid
  map_vars = ctx.get_map_variables()

  return all_func, inf_loss, inf_vars, map_loss, map_vars


def label_downsampler(n_labels, true_labels, n_extra_labels):
  lbls = tf.random_shuffle(tf.range(n_labels))[:n_extra_labels]
  lbls = tf.unique(tf.concat([lbls, true_labels], axis=0)).y
  return lambda logits: tf.gather(logits, lbls, axis=-1)

