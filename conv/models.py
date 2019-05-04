"""
Model functions. Will be invoked multiple times - to build train / inference
towers, and to implement model parallelism.
When building inference towers, TensorPack will set the reuse flag
"""
import tensorflow as tf
from ctx import *


@zs.meta_bayesian_net(reuse_variables=False)
def small_convnet(image, hps):
  bn = zs.BayesianNet()
  n_particles = hps.n_particles_per_dev
  image = tf.tile(image[None, ...], [n_particles, 1, 1, 1, 1])
  image = image / 4.0   # just to make range smaller
  keep_prob = tf.constant(0.5 if hps.dropout else 1.0)
  with BNNContext(bn, n_particles, hps.w_prior_std) as _bnn_ctx,\
      argscope(conv2d, activation=bn_relu, use_bias=False, kernel_size=3):
    logits = LinearWrap(image) \
      .conv2d('conv1.1', filters=64) \
      .conv2d('conv1.2', filters=64) \
      .max_pool('pool1', 3, strides=2, padding='same') \
      .conv2d('conv2.1', filters=128) \
      .conv2d('conv2.2', filters=128) \
      .max_pool('pool2', 3, strides=2, padding='same') \
      .conv2d('conv3.1', filters=128, padding='valid') \
      .conv2d('conv3.2', filters=128, padding='valid') \
      .flatten() \
      .dense('dense0', 1024 + 512, activation=tf.nn.relu) \
      .dropout(keep_prob) \
      .dense('dense1', 512, activation=tf.nn.relu) \
      .dense('linear', hps.classnum, activation=None)()
    logits = tf.nn.log_softmax(logits)
    bn.deterministic('logits', logits)
    bn.categorical('y', logits=logits)
    return bn, _bnn_ctx.param_names


def residual(name, l, increase_dim=False, first=False):
  shape = l.get_shape().as_list()
  in_channel = shape[2]

  if increase_dim:
    out_channel = in_channel * 2
    stride1 = (2, 2)
  else:
    out_channel = in_channel
    stride1 = (1, 1)

  with tf.variable_scope(name):
    b1 = l if first else bn_relu_std(l)
  c1 = conv2d(
    name + '/conv1', b1, out_channel, strides=stride1, activation=bn_relu_std)
  c2 = conv2d(
    name + '/conv2', c1, out_channel)
  if increase_dim:
    l = avg_pooling(l, 2)
    l = tf.pad(l, [
      [0, 0], [0, 0], [in_channel // 2, in_channel // 2], [0, 0], [0, 0]])
  l = c2 + l

  return l


@zs.meta_bayesian_net(reuse_variables=False)
def resnet(image, hps):
  bn = zs.BayesianNet()
  n_particles = hps.n_particles_per_dev
  image = tf.tile(image[None, ...], [n_particles, 1, 1, 1, 1])
  image = image / 128.0

  with BNNContext(bn, n_particles, hps.w_prior_std) as _bnn_ctx,\
      argscope(conv2d, use_bias=False, kernel_size=3):
    l = conv2d('conv0', image, 16, activation=bn_relu_std)
    l = residual('res1.0', l, first=True)
    for k in range(1, hps.n_res_block):
      l = residual('res1.{}'.format(k), l)
    # 32,c=16

    l = residual('res2.0', l, increase_dim=True)
    for k in range(1, hps.n_res_block):
      l = residual('res2.{}'.format(k), l)
    # 16,c=32

    l = residual('res3.0', l, increase_dim=True)
    for k in range(1, hps.n_res_block):
      l = residual('res3.{}'.format(k), l)
    l = bn_relu_std(l)
    # 8,c=64
    l = global_avg_pooling(l)
    #
    logits = tf.nn.log_softmax(dense('linear', l, 10))

  bn.deterministic('logits', logits)
  bn.categorical('y', logits=logits)
  return bn, _bnn_ctx.param_names


def rescale(inp, name):
  n_ch = inp.get_shape().as_list()[2]
  shape = [1, 1, n_ch, 1, 1]
  b = add_map_variable(name + '/beta', shape, tf.float32, tf.zeros_initializer())
  g = add_map_variable(name + '/gamma', shape, tf.float32, tf.ones_initializer())
  return (inp + b) * g

