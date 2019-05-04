import tensorflow as tf
import numpy as np
import zhusuan as zs


def init_bnn_weight(n_particles, n_in, n_out):
    w = np.random.normal(size=[n_particles, n_in, n_out]) / np.sqrt(
        n_in + n_out + 1)
    w = np.concatenate(
        [w, np.zeros((n_particles, 1, n_out))], axis=1).astype('f')
    return w


def layer(inp, rescale_weight, rf_sample, activation, linreg_w,
          n_particles=None):
    if n_particles is None:
        n_particles = inp.shape[0]
    inp = rescale_weight[:, None, :] * inp
    inp_bak = inp
    # inp -> rf imm
    inp = tf.concat([inp, tf.ones(tf.shape(inp)[:-1])[..., None]], -1)
    w_rf = tf.tile(rf_sample[None, ...], [n_particles, 1, 1])
    # nonlin
    n_rfs = int(rf_sample.shape[1])
    if activation == 'relu':
        imm = tf.nn.relu(inp @ w_rf) / np.sqrt(n_rfs)
    elif activation == 'cos':
        imm = tf.cos(inp @ w_rf) / np.sqrt(n_rfs / 2)
    # imm -> out. Prior p(w) must be N(0, I)
    imm = tf.concat([imm, tf.ones(tf.shape(imm)[:-1])[..., None]], -1)
    # out
    h = imm @ linreg_w
    return h


def sample_basis(activation, n_rfs, layer_sizes):
    rf_samples = []
    for i, n_in in enumerate(layer_sizes[:-1]):
        if activation == 'cos':
            assert n_rfs % 2 == 0
            rf_init = tf.random_normal([n_in, n_rfs//2], stddev=1)
            offset = -np.pi / 2 * tf.ones([1, n_rfs//2]) 
            rf_init = tf.concat(
                [tf.concat([rf_init, 0*offset], 0),
                 tf.concat([rf_init, 1*offset], 0)],
                axis=1)
        else:
            rf_init = tf.random_normal([n_in, n_rfs], stddev=1.)
            rf_init = tf.concat([rf_init, tf.zeros([1, n_rfs])], 0)
        rf_samples.append(tf.get_variable('rf_'+str(i), initializer=rf_init))
    return rf_samples

