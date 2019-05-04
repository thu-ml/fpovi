#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Based on `cifar-convnet.py` in Tensorpack examples, authored by Yuxin Wu
import tensorflow as tf
import argparse
import os

from tensorpack import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow import dataset

from ctx import *
from utils import replace_object_attr
from tp.training import *
import models, variationals


parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=str, default='-1')
parser.add_argument('-n_particles_per_dev', type=int, default=2)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-load', help='load model')
parser.add_argument('-variational', type=str, default='empirical')
parser.add_argument('-extra_batch_size', type=int, default=0)
# He suggests ~0.32*1.4; tfmodels uses 0.32
parser.add_argument('-w_prior_std', type=float, default=0.32)
parser.add_argument('-dropout', action='store_true')
parser.add_argument('-classnum', help='10 for cifar10 or 100 for cifar100',
                    type=int, default=10)
parser.add_argument('-model', type=str, default='resnet',
                    choices=['resnet', 'small_convnet'])
parser.add_argument('-n_res_block', type=int, default=5)
parser.add_argument('-prior_type', type=str, default='mm',
                    choices=['mm', 'fwdgrad'])
parser.add_argument('-mm_n_inp', type=int, default=2)
parser.add_argument('-mm_jitter', type=float, default=1)
parser.add_argument('--mm_n_downsample_class', '-mm_nc', type=int, default=4)
parser.add_argument('--mm_n_prior_towers', '-mm_npt', type=int, default=2)
parser.add_argument('-stein_method', type=str, default='gfsf',
                    choices=['svgd', 'gfsf'])
parser.add_argument('-max_epoch', type=int, default=400)
parser.set_defaults(dropout=False)


class Model(ModelDesc):
  def __init__(self, hps): 
    super(Model, self).__init__()
    self._hps = hps

  def inputs(self):
    D = 32 if self._hps.model == 'resnet' else 30
    return [tf.placeholder(tf.float32, (None, D, D, 3), 'input'),
            tf.placeholder(tf.int32, (None,), 'label')]

  def build_graph(self, image, label):
    hps = self._hps

    is_training = get_current_tower_context().is_training
    keep_prob = tf.constant(0.5 if is_training else 1.0)

    if is_training:
      tf.summary.image("train_image", image, 10)
    image = tf.transpose(image, [0, 3, 1, 2])

    if is_training and hps.extra_batch_size > 0:
      image_perturbed = image[-hps.extra_batch_size:]
      # pixel in 0..255; divide by sqrtN
      image_perturbed += tf.random_normal(tf.shape(image_perturbed)) * 1
      image = tf.concat([image[:-hps.extra_batch_size], image_perturbed], axis=0)

    if hps.model == 'resnet':
      build_meta_model = models.resnet
      variationals.default_initializer_mode = 'fan_out'
    else:
      build_meta_model = models.small_convnet

    def mm_builder(inp, n_particles, hps):
      nhps = replace_object_attr(hps, 'n_particles_per_dev', n_particles)
      return build_meta_model(inp, nhps)

    var_func = {
      'empirical': variationals.empirical,
      'f_svgd': variationals.f_svgd
    }[hps.variational]

    obs_weight = tf.to_float(hps.n_train) / tf.to_float(tf.shape(image)[0])
    with tf.variable_scope('variational'):
      var_logits, _inf_loss, self.inf_vars, _map_loss, self.map_vars = var_func(
        mm_builder, image, {'y': label}, obs_weight, devices=hps.devices,
        hps=hps)

    # downscale training signals so prev hps can apply
    self.inf_loss = tf.reduce_sum(_inf_loss, name='cost') / float(hps.n_train)
    self.map_loss = _map_loss / float(hps.n_train)

    pred_probs = tf.clip_by_value(
      tf.reduce_mean(tf.nn.softmax(var_logits, axis=-1), axis=0),
      1e-7, 1 - 1e-7)
    # monitor training error
    correct = tf.to_float(
      tf.nn.in_top_k(pred_probs, label, 1), name='correct')
    add_moving_summary(tf.reduce_mean(correct, name='accuracy'))
    log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=tf.log(pred_probs), labels=label)
    add_moving_summary(tf.reduce_mean(log_prob, name='log_prob'))
    add_param_summary(('.*/W', ['histogram']))   # monitor W

  def optimizer(self):
    lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
    tf.summary.scalar('lr', lr)
    if self._hps.model == 'resnet':
      return tf.train.MomentumOptimizer(lr, 0.9)
    else:
      return tf.train.AdamOptimizer(lr, epsilon=1e-3)


def get_data(train_or_test, args):
  isTrain = train_or_test == 'train'
  if args.classnum == 10:
    ds = dataset.Cifar10(train_or_test)
  else:
    ds = dataset.Cifar100(train_or_test)
  data_size = ds.size()
  pp_mean = ds.get_per_pixel_mean()
  if args.model == 'resnet':
    if isTrain:
      augmentors = [
        imgaug.CenterPaste((40, 40)),
        imgaug.RandomCrop((32, 32)),
        imgaug.Flip(horiz=True),
        imgaug.MapImage(lambda x: x - pp_mean),
      ]
    else:
      augmentors = [
        imgaug.MapImage(lambda x: x - pp_mean)
      ]
  else:
    if isTrain:
      augmentors = [
        imgaug.RandomCrop((30, 30)),
        imgaug.Flip(horiz=True),
        imgaug.Brightness(63),
        imgaug.Contrast((0.2, 1.8)),
        imgaug.MeanVarianceNormalize(all_channel=True)
      ]
    else:
      augmentors = [
        imgaug.CenterCrop((30, 30)),
        imgaug.MeanVarianceNormalize(all_channel=True)
      ]
  ds = AugmentImageComponent(ds, augmentors)
  ds = BatchData(ds, args.batch_size, remainder=not isTrain)
  # ds = BatchData(ds, 16, remainder=not isTrain)
  if isTrain:
    ds = PrefetchDataZMQ(ds, 5)
  return ds, data_size


def get_config(args):
  # prepare dataset
  dataset_train, n_train = get_data('train', args)
  dataset_test, _ = get_data('test', args)

  # tf config
  tf_config = tf.ConfigProto(allow_soft_placement=True)
  tf_config.gpu_options.allow_growth = True

  # update args
  from tensorflow.python.client import device_lib
  devices = [d for d in device_lib.list_local_devices() if d.device_type=='GPU']
  setattr(args, 'devices', devices)
  setattr(args, 'n_train', n_train)

  if args.model == 'resnet':
    lr_setter = ScheduledHyperParamSetter(
      'learning_rate',
      [(1, 0.1), (82, 0.01), (123, 0.001), (300, 0.0002)])
  else:
    def lr_func(lr):
      if lr < 3e-7:
        raise StopTraining()
      return lr * 0.31
    lr_setter = StatMonitorParamSetter(
      'learning_rate', 'validation_accuracy', lr_func,
      threshold=0.001, last_k=10, reverse=True)

  return TrainConfig(
    model=Model(args),
    dataflow=dataset_train,
    session_config=tf_config,
    callbacks=[
      ModelSaver(),
      InferenceRunner(dataset_test, ScalarStats(['accuracy', 'log_prob'])),
      lr_setter,
    ],
    max_epoch=args.max_epoch,
  )


if __name__ == '__main__':
  args = parser.parse_args()

  if args.gpu != '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

  import datetime
  dt = datetime.datetime.now()
  timestr = '{}-{}-{}-{}'.format(dt.day, dt.hour, dt.minute, dt.second)
  logdir = '/tmp/train_log/cifar_{}_{}'.format(args.classnum, args.variational)
  logdir = os.path.join(logdir, timestr)

  nr_gpu = len(args.gpu.split(','))

  with tf.Graph().as_default():
    logger.set_logger_dir(logdir)
    config = get_config(args)
    if args.load:
      config.session_init = SaverRestore(args.load)

    launch_train_with_config(config)

