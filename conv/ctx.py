# Context manager for BNN
# To infer subsequent layer shapes, you must have a tensor to work on. So we
# can't avoid wasting time on constructing the "prior bn".
# With the prior bn, we can monkeypatch get_variables, so that when it knows
# we are building a BNN metabn, it replaces get_variable calls w/ bn.stochastic.
# 

# Inspired by TensorPack

import sys
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
import zhusuan as zs
from tensorpack.utils import logger
from tensorpack import get_current_tower_context
from tp.batch_norm import BatchNorm


_current_ctx = None
_current_layer_name = None
_current_argscopes = []
layer_names = []
unnamed_layer_names = []
do_log_shape = True
_map_var_dict = {}


def add_map_variable(name, shape, dtype, initializer):
    g = tf.get_default_graph()
    if g not in _map_var_dict:
        _map_var_dict[g] = {}
    if name not in _map_var_dict[g]:
        with tf.device('/cpu:0'):
            nv = tf.get_variable(name, shape, dtype, initializer)
        _map_var_dict[g][name] = nv
    return tf.identity(_map_var_dict[g][name])


def get_map_variables():
    g = tf.get_default_graph()
    if g not in _map_var_dict:
        _map_var_dict[g] = {}
    return list(_map_var_dict[g].values())


class BNNContext:

    def __init__(self, bn, n_particles, w_prior_std):
        self.bn = bn
        self.n_particles = n_particles
        self.param_names = []
        self.w_prior_std = w_prior_std

    def __enter__(self):
        global _current_ctx
        assert _current_ctx is None
        _current_ctx = self
        return self

    def __exit__(self, *args, **kw):
        global _current_ctx
        _current_ctx = None
        return False

    def get_prior_std(self, tag):
        if tag == 'dense/W' or tag == 'conv2d/kernel':
            return self.w_prior_std
        else:
            return 1e4

    def get_prior_rv(self, name, shape, tag):
        # by default, use non-informative prior for everything
        name = _current_layer_name + '/' + name
        self.param_names.append((name, tag))
        std = self.get_prior_std(tag)
        return self.bn.normal(
            name=name, mean=tf.zeros(shape), std=tf.ones(shape) * std,
            n_samples=self.n_particles, group_ndims=len(shape))


def log_shape(name, inp, out):
    if do_log_shape:
        logger.info("layer {}: {} -> {}".format(
            name, inp.get_shape().as_list(), out.get_shape().as_list()))


def set_shape_logger(val):
    global do_log_shape
    do_log_shape = val


@contextmanager
def argscope(func_or_funcs, **kw):
    if not isinstance(func_or_funcs, list):
        func_or_funcs = [func_or_funcs]
    _current_argscopes.append([func_or_funcs, kw])
    yield
    _current_argscopes.pop()


def wrap_layer(fn):
    layer_names.append(fn.__name__)
    def wrapped(*args, **kw):
        global _current_layer_name, _current_argscopes
        # Record layer name for ctx
        _current_layer_name = args[0]
        args = args[1:]
        # Apply argscopes
        for funcs, kw_to_add in reversed(_current_argscopes):
            if wrapped not in funcs:
                continue
            for k in kw_to_add:
                if k not in kw:
                    kw[k] = kw_to_add[k]
        # Call fn in new scope
        with tf.variable_scope(_current_layer_name):
            ret = fn(*args, **kw)
        log_shape(_current_layer_name, args[0], ret)
        # Pop stacks
        _current_layer_name = None
        return ret
    return wrapped


def register_unnamed_layer(fn):
    unnamed_layer_names.append(fn.__name__)
    return fn


class LinearWrap:

    def __getattr__(self, fname):
        fn = globals()[fname]
        def method(*args, **kw):
            # if fname not in unnamed_layer_names:
            #     kw['name'] = args[0]
            #     args = args[1:]
            if fname in layer_names:
                args = [args[0], self.inp] + list(args[1:])
            return LinearWrap(fn(*args, **kw))
        return method

    def __init__(self, inp):
        self.inp = inp

    def __call__(self):
        return self.inp


def to_2d_list(ksize):
    if isinstance(ksize, int):
        ksize = [ksize, ksize]
    else:
        ksize = list(ksize)
        assert len(ksize) == 2
    return ksize


def to_4d_list(ksize):
    return [1, 1] + to_2d_list(ksize)


@register_unnamed_layer
def flatten(inp):
    n_out = np.prod(inp.get_shape().as_list()[2:])
    return tf.reshape(inp, [tf.shape(inp)[0], tf.shape(inp)[1], n_out])


@wrap_layer
def max_pool(inp, ksize, strides, padding):
    ctx = _current_ctx
    ksize = to_4d_list(ksize)
    strides = to_4d_list(strides)
    _, _, c, h, w = inp.get_shape().as_list()
    batch_size = tf.shape(inp)[1]
    inp = tf.reshape(inp, [-1, c, h, w])
    ret = tf.nn.max_pool(
        inp, ksize, strides, padding.upper(), data_format='NCHW')
    orig_shape = [ctx.n_particles, batch_size] + ret.get_shape().as_list()[1:]
    return tf.reshape(ret, orig_shape)


@register_unnamed_layer
def dropout(inp, keep_prob):
    is_training = get_current_tower_context().is_training
    return tf.layers.dropout(inp, rate=1 - keep_prob, training=is_training)


@register_unnamed_layer
def global_avg_pooling(inp):
    assert len(inp.get_shape().as_list()) == 5
    return tf.reduce_mean(inp, [3, 4])


@register_unnamed_layer
def avg_pooling(inp, pool_size):
    pool_size = [int(pool_size), pool_size]
    strides = pool_size
    # 5d -> 4d
    _, _, c, h, w = inp.get_shape().as_list()
    out = tf.reshape(inp, [-1, c, h, w])
    #
    out = tf.layers.average_pooling2d(
        out, pool_size, strides, data_format='channels_first')
    # 4d -> 5d
    _, c, h, w = out.get_shape().as_list()
    n_particles = _current_ctx.n_particles
    out = tf.reshape(out, [n_particles, -1, c, h, w])
    log_shape('avg_pool', inp, out)
    return out


@wrap_layer
def dense(inp, n_out, activation=None):
    ctx = _current_ctx
    if activation is None:
        activation = tf.identity
    _, _, n_in = inp.get_shape().as_list()
    # Will return w/ extra n_particles dim
    W = ctx.get_prior_rv(name='W', shape=[n_in, n_out], tag='dense/W')
    b = ctx.get_prior_rv(name='b', shape=[n_out], tag='dense/b')
    return activation(inp @ W + b[:, None, :])


@wrap_layer
def conv2d(inp, filters, kernel_size, strides=(1, 1), padding='same',
           use_bias=True, activation=None):
    # Convert all args. Assume NCHW
    ctx = _current_ctx
    if activation is None:
        activation = tf.identity
    _, _, in_channels, in_height, in_width = inp.get_shape().as_list()
    kernel_shape = to_2d_list(kernel_size) + [in_channels, filters]
    strides = [1, 1] + list(strides)
    padding = padding.upper()

    kernel = ctx.get_prior_rv(
        name='kernel', shape=kernel_shape, tag='conv2d/kernel')
    if use_bias:
        b = ctx.get_prior_rv(name='b', shape=[filters], tag='conv2d/b')

    def conv_fn(i):
        ret = tf.nn.conv2d(
            inp[i], kernel[i], strides, padding, data_format='NCHW')
        if use_bias:
            ret += tf.reshape(b[i], [1, 1, 1, filters])
        return ret

    if isinstance(ctx.n_particles, int):
        ret = tf.concat(
            [tf.expand_dims(conv_fn(i), 0) for i in range(ctx.n_particles)],
            axis=0)
    else:
        ret = tf.map_fn(conv_fn, tf.range(ctx.n_particles), dtype=tf.float32)
    return activation(ret)


# batch_norm and relu. intended to be used as activation, and scope will be
# handled by the previous layer
def bn_relu(inp):
    with tf.variable_scope('bn') as scope:
        inp = BatchNorm(inp, axis=[0, 2], center=False, scale=False)
    return tf.nn.relu(inp)


def bn_relu_std(inp):
    with tf.variable_scope('bn') as scope:
        inp = BatchNorm(inp, axis=[0, 2], center=False, scale=False)
    return tf.nn.relu(inp)

