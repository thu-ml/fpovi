#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os

import tensorflow as tf
from six.moves import range, zip
import numpy as np
import zhusuan as zs
import experiments.utils
from experiments import wrapped_supervisor

import dataset
from rf import init_bnn_weight
from utils import inv_softplus


parser = experiments.utils.parser('dgp')
parser.add_argument('-layers', nargs='+', default=[50], type=int)
parser.add_argument('-n_particles', default=5, type=int)
parser.add_argument('-batch_size', default=1000, type=int)
parser.add_argument('-n_epoch', default=5000, type=int)
parser.add_argument('-dataset', default='concrete', type=str)
parser.add_argument('-lr', default=1e-2, type=float)
parser.add_argument('-seed', default=1234, type=int)
parser.add_argument('-test_freq', default=50, type=int)
parser.add_argument('-dump_freq', default=200, type=int)
parser.add_argument('-fix_variance', default=-1, type=float)
parser.add_argument('-data_dir',
                    default=os.path.join(
                        experiments.utils.source_dir(), '../data'),
                    type=str)
parser.add_argument('-dump_pred_dir', default='/tmp/pred-last-hmc.bin', type=str)


@zs.meta_bayesian_net(scope="bnn", reuse_variables=True)
def build_model(x, layer_sizes, n_particles, fix_variance):
    """
    :param x: [batch_size, n_dims] (?)
    :param layer_sizes: list of length n_layers+1
    """
    assert int(x.shape[1]) == layer_sizes[0]

    bn = zs.BayesianNet()
    h = tf.tile(x[None, ...], [n_particles, 1, 1])
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        std = np.sqrt(1 / n_out)
        w = bn.normal("w" + str(i),
                      tf.zeros([n_in+1, n_out]),
                      std=std.astype('f'),
                      group_ndims=2, n_samples=n_particles)
        h = tf.concat([h, tf.ones(tf.shape(h)[:-1])[..., None]], -1) @ w
        if i != len(layer_sizes) - 2:
            h = tf.nn.relu(h)

    assert h.get_shape().as_list()[-1] == 1
    h = tf.squeeze(h, -1)
    y_mean = bn.deterministic("y_mean", h)
    y_std = bn.inverse_gamma(
        "y_std", alpha=[1.], beta=[0.1], n_samples=n_particles, group_ndims=1)
    assert len(y_std.shape) == 2
    y_logstd = tf.get_variable("y_logstd", shape=[],
                               initializer=tf.constant_initializer(inv_softplus(.5)))
    bn.y_logstd = y_logstd
    y_std_sym = tf.nn.softplus(y_logstd)
    if fix_variance > 0:
        y_std_sym = tf.clip_by_value(y_std_sym, fix_variance, fix_variance + 1e-5)
    y = bn.normal("y", y_mean, std=y_std_sym)

    return bn


def main(hps):
    tf.set_random_seed(hps.seed)
    np.random.seed(hps.seed)

    # Load data
    data_path = os.path.join(hps.data_dir, hps.dataset + '.data')
    data_func = dataset.data_dict()[hps.dataset]
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_func(data_path)
    x_train = np.vstack([x_train, x_valid])
    y_train = np.hstack([y_train, y_valid])
    n_train, x_dim = x_train.shape
    x_train, x_test, mean_x_train, std_x_train = dataset.standardize(
        x_train, x_test)
    y_train, y_test, mean_y_train, std_y_train = dataset.standardize(
        y_train, y_test)

    # Define model parameters
    n_hiddens = hps.layers 

    # Build the computation graph
    x = tf.placeholder(tf.float32, shape=[None, x_dim])
    y = tf.placeholder(tf.float32, shape=[None])
    layer_sizes = [x_dim] + n_hiddens + [1]
    w_names = ["w" + str(i) for i in range(len(layer_sizes) - 1)]

    meta_model = build_model(x, layer_sizes, hps.n_particles, hps.fix_variance)

    def log_joint(bn):
        log_pws = bn.cond_log_prob(w_names)
        log_py_xw = bn.cond_log_prob('y')
        return tf.add_n(log_pws) + tf.reduce_mean(log_py_xw, 1) * n_train

    meta_model.log_joint = log_joint

    latent = {}
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        buf = tf.get_variable(
            'buf_'+str(i),
            initializer=init_bnn_weight(hps.n_particles, n_in, n_out))
        latent['w'+str(i)] = buf

    hmc = zs.HMC(step_size=hps.lr, n_leapfrogs=10, adapt_step_size=True)
    sample_op, hmc_info = hmc.sample(meta_model, observed={'y': y}, latent=latent)

    var_bn = meta_model.observe(**latent)
    log_joint = var_bn.log_joint()
    optimizer = tf.train.AdamOptimizer(learning_rate=hps.lr)
    global_step = tf.get_variable(
        'global_step', initializer=0, trainable=False)
    opt_op = optimizer.minimize(
        -log_joint, var_list=[var_bn.y_logstd], global_step=global_step)

    # prediction: rmse & log likelihood
    y_mean = var_bn["y_mean"]
    y_pred = tf.reduce_mean(y_mean, 0)
    rmse = tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) * std_y_train
    log_py_xw = var_bn.cond_log_prob("y")
    log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0)) - \
        tf.log(std_y_train)
    ystd_avg = var_bn.y_logstd

    # Define training/evaluation parameters
    epochs = hps.n_epoch
    batch_size = hps.batch_size
    iters = int(np.ceil(x_train.shape[0] / float(batch_size)))
    test_freq = hps.test_freq

    # Run the inference
    dump_buf = []
    with wrapped_supervisor.create_sv(hps, global_step=global_step) as sv:
        sess = sv.sess_
        for epoch in range(1, epochs + 1):
            lbs = []
            perm = np.arange(x_train.shape[0])
            np.random.shuffle(perm)
            x_train = x_train[perm]
            y_train = y_train[perm]
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, _, accr = sess.run(
                    [sample_op, opt_op, hmc_info.acceptance_rate],
                    feed_dict={x: x_batch, y: y_batch})
                lbs.append(accr)
            if epoch % 10 == 0:
                print('Epoch {}: Acceptance rate = {}'.format(epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                test_rmse, test_ll = sess.run(
                    [rmse, log_likelihood],
                    feed_dict={x: x_test, y: y_test})
                print('>> TEST')
                print('>> Test rmse = {}, log_likelihood = {}'
                      .format(test_rmse, test_ll))

            if epoch>epochs //3 and epoch % hps.dump_freq == 0:
                dump_buf.append(sess.run(var_bn['y_mean'], {x:x_test, y: y_test}))

        if len(hps.dump_pred_dir) > 0:
            pred_out = sess.run([var_bn['y_mean'], var_bn.y_logstd], {x: x_test, y: y_test})
            pred_out[0] = np.concatenate(dump_buf, axis=0)
            pred_out[0] = pred_out[0] * std_y_train + mean_y_train
            pred_out[1] = np.exp(pred_out[1])
            f = lambda a, b: [a*std_x_train + mean_x_train, b*std_y_train + mean_y_train]
            todump = pred_out + f(x_test, y_test) + f(x_train, y_train)
            with open(hps.dump_pred_dir, 'wb') as fout:
                import pickle
                pickle.dump(todump, fout)


if __name__ == "__main__":
    hps = parser.parse_args()
    experiments.utils.preflight(hps)
    main(hps)
