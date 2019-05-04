import tensorflow as tf
import zhusuan as zs
import numpy as np


def add_namescope(f):
    def wrapped(*inp, **kw):
        with tf.variable_scope(kw['name']):
            return f(*inp, **kw)
    return wrapped


def he_init(name, n_particles, n_in, n_out, with_bias=False):
    init = np.random.normal(size=[n_particles, n_in, n_out])
    # init /= np.sqrt((n_in + 1) / 2)
    if with_bias:
        init = np.concatenate(
            [init, np.zeros((n_particles, 1, n_out))], axis=1)
    init = init.astype('f')
    w = tf.get_variable(name, initializer=tf.constant(init))
    d = zs.distributions.Normal(tf.zeros(init.shape[1:]), std=1., group_ndims=2)
    return w, d.log_prob(w)


def concat_ones(imm):
    return tf.concat([imm, tf.ones(tf.shape(imm)[:-1])[..., None]], -1)


def inv_softplus(a):
    return np.log(np.exp(a) - 1)


def get_positive_variable(name, shape, init_val):
    init_val = init_val
    buf = tf.get_variable(
        name, shape=shape,
        initializer=tf.constant_initializer(inv_softplus(init_val)))
    return tf.nn.softplus(buf)


@add_namescope
def conv2d_rf(inp, kernel_size, n_out, use_bias=False,
              strides=(1, 1), padding='SAME',
              rf=None, prior_list=None, n_particles=None, n_rfs=None, name=None):
    """
    :param inp: NHWC
    :param kernel_basis: [height, width, n_in, n_out]
    """
    strides = [1] + list(strides) + [1]
    _, h, w, n_in = inp.get_shape().as_list()
    batch_size = tf.shape(inp)[0]
    assert n_in % n_particles == 0
    n_in = n_in // n_particles

    # rf layer
    kernel_init = tf.reshape(
        rf.sample_basis(n_in * kernel_size * kernel_size, n_rfs),
        [kernel_size, kernel_size, n_in, n_rfs])
    rf_filter = tf.get_variable(
        'rf_kernel', initializer=kernel_init, trainable=False)
    scale = get_positive_variable(
        'rf_scale', [n_particles, kernel_size, kernel_size, n_in],
        # 1)
        2 / kernel_size / np.sqrt(n_in))
    # gp layer
    w_out, pw = he_init('w', n_particles, n_rfs, n_out, with_bias=use_bias)
    prior_list.append(pw)
    inps = tf.split(inp, n_particles, axis=3)
    outs = []
    for i, inp in enumerate(inps):
        filter_i = rf_filter * tf.tile(scale[i][..., None], [1, 1, 1, n_rfs])
        imm = tf.nn.conv2d(inp, filter_i, strides, padding, data_format='NHWC')
        imm = rf.activation(imm, n_rfs)
        if use_bias:
            imm = tf.concat_ones(imm)
        #
        _, h1, _, _ = imm.get_shape().as_list()
        w_cur = tf.tile(tf.reshape(w_out[i], [1, 1, -1, n_out]),
                        [batch_size, h1, 1, 1])
        out = imm @ w_cur
        outs.append(out)
    out = tf.concat(outs, 3)
    print('Layer {}: {} -> {} '.format(
        name, inp.get_shape().as_list(), out.get_shape().as_list()))
    return out


@add_namescope
def fc_rf(inp, n_out,
          n_particles=None, n_rfs=None, rf=None, prior_list=None, name=None):
    """
    :param inp: [n_particles, batch_size, n_in]
    """
    n_particles, _, n_in = inp.get_shape().as_list()
    scale = get_positive_variable(
        'rf_scale', [n_particles, 1, n_in], 2 / np.sqrt(n_in))
    inp = scale * inp
    rf_sample = tf.get_variable(
        'rf_w', initializer=rf.sample_basis(n_in+1, n_rfs), trainable=False)
    w_rf = tf.tile(rf_sample[None, ...], [n_particles, 1, 1])
    imm = rf.activation(concat_ones(inp) @ w_rf, n_rfs)
    w_out, pw = he_init('w', n_particles, n_rfs, n_out, with_bias=True)
    prior_list.append(pw)
    h = concat_ones(imm) @ w_out
    return h
    

# def conv2d_grouped(inp, kernel_size, name, activation=None, bn=None):
#     """
#     :param inp: NCHW
#     :param kernel: [n_particles, height, width, n_in, n_out]
#     """
#     in_channel = int(inp.get_shape().as_list()[1])
#     assert in_channel % n_particles == 0
#     return tf.nn.conv2d()
# 
# def fc_grouped(inp, n_out, name, activation=None, bn=None):
#     """
#     :param inp: [batch_size, n_particles, n_in]
#     """
#     assert len(inp.get_shape().as_list()) == 3
#     n_particles, n_in = inp.get_shape().as_list()[1:]
#     w = bn.normal('W_' + name, tf.zeros([n_in, n_out]), std=1.,
#                   group_ndims=2, n_samples=n_particles)
#     b = bn.normal('b_' + name, tf.zeros([1, n_out]), std=1.,
#                   group_ndims=2, n_samples=n_particles)
#     inp = tf.concat(

