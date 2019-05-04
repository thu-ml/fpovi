import os
import sys
import numpy as np
import tensorflow as tf
from scipy.misc import logsumexp
from sklearn.cluster import KMeans


def merge_last_axes(tensor, ax_start, name=None):
  shape = tensor.get_shape().as_list()
  for i, s in enumerate(shape):
    if s is None:
      shape[i] = tf.shape(tensor)[i]
  new_shape = shape[:ax_start] + [-1]
  return tf.reshape(tensor, new_shape, name=name)


def replace_object_attr(obj, key, val):
  dct = obj.__dict__.copy()
  dct[key] = val
  class A:
    pass
  ret = A()
  for k in dct:
    setattr(ret, k, dct[k])
  return ret


def matrix_inverse(m, eig_min, eig_max=1e100): 
  m = tf.to_double(m)
  eig_min = tf.to_double(eig_min)
  uu, vv = tf.self_adjoint_eig(m)
  uu = 1. / tf.clip_by_value(uu, eig_min, eig_max)
  return vv @ tf.matrix_diag(uu) @ tf.transpose(vv)


def mvn_log_prob(obs, prec, mean):
  dst = obs - mean[None, ...]
  return tf.matrix_diag_part(-1/2 * dst @ prec @ tf.transpose(dst))


def reduce_moments_ax0(t):
  t = tf.to_double(t)
  n = tf.to_double(tf.shape(t)[0])
  mean = tf.reduce_mean(t, axis=0, keepdims=True)
  cov = tf.transpose(t - mean) @ (t - mean) / (n - 1)
  return tf.squeeze(mean, axis=0), cov


def add_perturb_input(x, extra_batch_size, n_train, ptb_scale=0.1, ptb_type='kde'):
  # assuming standardized real-value input
  if extra_batch_size == 0:
    return x, None

  x_dim = int(x.shape[1])
  if extra_batch_size > 0:
    x_extra_ph = None
    indices = tf.random_uniform( 
      [extra_batch_size], 0, tf.shape(x)[0], dtype=tf.int32)
    to_perturb = tf.gather(x, indices, axis=0)
  else:
    to_perturb = x_extra_ph = tf.placeholder_with_default(
      tf.zeros([-extra_batch_size, x_dim], dtype=tf.float32),
      [None, x_dim], 'x_extra_base')

  ptb_size = tf.shape(to_perturb)[0]
  if ptb_type == 'kde': 
    perturb_scale = ptb_scale / np.sqrt(n_train * x_dim)
    x_perturbed = to_perturb + tf.random_normal(
      [ptb_size, x_dim], stddev=perturb_scale)
  else:
    indices = tf.random_uniform([ptb_size], 0, ptb_size, dtype=tf.int32)
    k = tf.random_uniform([ptb_size], 0, 1)[:, None]
    x_perturbed = to_perturb * k + tf.gather(to_perturb, indices) * (1 - k)
    
  model_inp = tf.stop_gradient(tf.concat([x, x_perturbed], axis=0))
  return model_inp, x_extra_ph


def inv_softplus(a):
  return np.log(np.exp(a) - 1)


def kl_mvn(qd, pd): # KL(q||p)
  n_channels = int(pd.cov_tril.shape[0])
  n_dims = int(pd.cov_tril.shape[1])
  print(n_channels, n_dims)
  assert len(pd.cov_tril.shape) == 3 and \
    list(qd.cov_tril.shape) == [n_channels, n_dims, n_dims]
  # shape == [n_out]
  log_det_p = 2 * tf.reduce_sum(
    tf.log(tf.matrix_diag_part(pd.cov_tril)), axis=-1)
  log_det_q = 2 * tf.reduce_sum(
    tf.log(tf.matrix_diag_part(qd.cov_tril)), axis=-1)
  # LL^T = cov, inv(LT) inv(L) = inv(cov)
  p_cov_tril_inv = tf.matrix_triangular_solve(
    pd.cov_tril, 
    tf.eye(n_dims, batch_shape=[n_channels]))
  cov_p_inv = tf.matrix_transpose(p_cov_tril_inv) @ \
    p_cov_tril_inv
  cov_q = qd.cov_tril @ tf.matrix_transpose(qd.cov_tril)
  mean_p_minus_q = tf.expand_dims(pd.mean - qd.mean, -1)
  klqp = 0.5 * (
    log_det_p - log_det_q - tf.cast(n_dims, tf.float32) + 
    tf.reduce_sum(tf.matrix_diag_part(cov_p_inv @ cov_q),
            axis=-1) +
    tf.squeeze(tf.matrix_transpose(mean_p_minus_q) @ \
           cov_p_inv @ mean_p_minus_q)
  )
  assert len(klqp.shape) == 1
  return tf.reduce_sum(klqp)


class MCBuffer:

  def __init__(self):
    self.log_lhood = []
    self.predicted = []

  def add_sample(self, log_lhood, predicted, y_true):
    """
    :param log_lhood: array of shape (n_chains,)
    :param predicted: array of shape (n_chains, y_true.shape[0])
    """
    assert list(log_lhood.shape) == [predicted.shape[0]]
    assert list(y_true.shape) == [predicted.shape[1]]
    self.log_lhood.append(log_lhood)
    self.predicted.append(predicted)
    lh = np.array(self.log_lhood).squeeze()
    est_lh = logsumexp(lh) - np.log(lh.shape[0])
    pred = np.mean(
      np.array(self.predicted).reshape((-1, y_true.shape[0])),
      axis=0)
    rmse = np.sqrt(np.mean((pred - y_true) ** 2))
    return est_lh, rmse


def kmeans(x_train, n_z):
  md = KMeans(n_clusters=n_z)
  if x_train.shape[0] < 1000:
    inp = x_train
  else:
    inp = x_train[np.random.choice(x_train.shape[0], 1000)]
  md.fit(inp)
  return np.array(md.cluster_centers_).astype('f')


def load_data(hps):
  from examples.utils import dataset
  from examples import conf
  data_path = os.path.join(conf.data_dir, hps.dataset + '.data')
  data_func = getattr(dataset, 'load_uci_' + hps.dataset)
  x_train, y_train, x_valid, y_valid, x_test, y_test = data_func(data_path)
  x_train = np.vstack([x_train, x_valid])
  y_train = np.hstack([y_train, y_valid])
  n_train, hps.n_x = x_train.shape
  x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
  y_train, y_test, mean_y_train, std_y_train = dataset.standardize(
    y_train, y_test)
  return x_train, y_train, x_test, y_test, n_train, mean_y_train, std_y_train


def var(name):
  rets = []
  for var in tf.global_variables():
    if var.name.find(name) != -1:
      rets.append(var)
  if len(rets) == 1:
    return rets[0]
  return rets


def op(name):
  rets = [n.name for n in tf.get_default_graph().as_graph_def().node \
      if n.name.find(name) != -1]
  rets = [tf.get_default_graph().get_operation_by_name(n) for n in rets]
  if len(rets) == 1:
    return rets[0]
  return rets


def fwd_gradients(ys, xs, d_xs):
  """Forward-mode pushforward analogous to the pullback defined by tf.gradients.
  With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
  the vector being pushed forward."""
  v = tf.placeholder_with_default(tf.zeros_like(ys), shape=ys.get_shape())  # dummy variable
  g = tf.gradients(ys, xs, grad_ys=v)
  return tf.gradients(g, v, grad_ys=d_xs)


def train_loop(sess, hps, x_train, y_train, x_test, y_test,
         infer_op, rmse_op, loglh_op, ystd_op,
         x_ph, y_ph, x_extra_ph=None, svgd_profile=None,
         x_valid=None, y_valid=None):

  iters = int(np.ceil(x_train.shape[0] / float(hps.batch_size)))
  best_val_ll = -1e100
  for epoch in range(1, hps.n_epoch + 1):
    lbs = []
    rmses = []
    perm = np.arange(x_train.shape[0])
    np.random.shuffle(perm)
    x_train = x_train[perm]
    y_train = y_train[perm]
    for t in range(iters):
      x_batch = x_train[t * hps.batch_size:(t + 1) * hps.batch_size]
      y_batch = y_train[t * hps.batch_size:(t + 1) * hps.batch_size]
      fd = {
        x_ph: x_batch,
        y_ph: y_batch
      }
      if hasattr(hps, 'extra_batch_size') and hps.extra_batch_size < 0:
        t0 = np.random.randint(iters)
        bsz = abs(hps.extra_batch_size)
        ex_batch = x_train[t0*bsz: (t0+1)*bsz]
        fd[x_extra_ph] = ex_batch
      _, rmse, lb = sess.run([infer_op, rmse_op, loglh_op], fd)
      assert not np.isnan(lb)
      rmses.append(rmse)
      lbs.append(lb)
    if x_train.shape[0] > 1000 or epoch % 10 == 0:
      print('Epoch {}: rmse = {}, Log likelihood = {}'.format(
        epoch, np.mean(rmses), np.mean(lbs)))

    if epoch % hps.test_freq == 0:
      #
      test_rmse, test_ll, test_avgstd = sess.run(
        [rmse_op, loglh_op, ystd_op],
        feed_dict={x_ph: x_test, y_ph: y_test})
      print('>> TEST')
      print('>> Test rmse = {}, log_likelihood = {} avg std = {}'
          .format(test_rmse, test_ll, test_avgstd))

      if x_valid is not None:
        val_rmse, val_ll = sess.run(
          [rmse_op, loglh_op],
          feed_dict={x_ph: x_valid, y_ph: y_valid})
        print('>> Valid rmse = {}, log_likelihood = {}'.format(val_rmse, val_ll))
        if val_ll > best_val_ll:
          best_val_ll = val_ll
          best_metrics = (test_rmse, test_ll, test_avgstd, best_val_ll)

      if hasattr(hps, 'profile_svgd') and hps.profile_svgd:
        pvals = sess.run(svgd_profile[1], fd)
        print('>> Profile\n{}'.format('\n'.join(
          ['{}: {}'.format(k, v)
           for k, v in zip(svgd_profile[0], pvals)])))

  if x_valid is not None:
    print('>> TEST RESULT')
    print('>> Test rmse = {}, log_likelihood = {} avg std = {} best valid ll = {}'.format(
      *best_metrics))

