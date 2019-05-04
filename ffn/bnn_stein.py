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
from rf import init_bnn_weight
from svgd import stein_variational_gradient_stationary
from utils import inv_softplus, train_loop
from bnn_stein_f import load_data


parser = experiments.utils.parser('bnn0')
parser.add_argument('-layers', nargs='+', default=[50], type=int)
parser.add_argument('-n_particles', default=20, type=int)
parser.add_argument('-batch_size', default=100, type=int)
parser.add_argument('-n_epoch', default=5000, type=int)
parser.add_argument('-dataset', default='concrete', type=str)
parser.add_argument('-lr', default=1e-2, type=float)
parser.add_argument('-optimizer', default='adam', type=str)
parser.add_argument('-seed', default=1234, type=int)
parser.add_argument('-test_freq', default=50, type=int)
parser.add_argument('-fix_variance', default=-1, type=float)
parser.add_argument('-valid', action='store_true', dest='use_valid')
parser.add_argument('-dump_pred_dir', default='/tmp/pred-last-bn.bin', type=str)
parser.add_argument('-data_dir',
                    default=os.path.join(
                        experiments.utils.source_dir(), '../data'),
                    type=str)
parser.add_argument('-model_spec', default='normal', type=str,
                    choices=['normal', 'lq'])
parser.add_argument('-psvi_method', default='svgd', type=str,
                    choices=['svgd', 'gfsf', 'wsgld', 'pisgld'])
parser.add_argument('-logits_w_sd', default=40, type=float)
parser.add_argument('-save', action='store_true', dest='save_model')
parser.set_defaults(use_valid=False, regression=True, save_model=False)


@zs.meta_bayesian_net(scope="bnn", reuse_variables=True)
def build_model(x, layer_sizes, n_particles, model_type, is_regression,
                logits_w_sd):
    """
    :param x: [batch_size, n_dims] (?)
    :param layer_sizes: list of length n_layers+1
    """
    assert int(x.shape[1]) == layer_sizes[0]

    bn = zs.BayesianNet()
    h = tf.tile(x[None, ...], [n_particles, 1, 1])
    if model_type == 'lq':
        w_scale = bn.inverse_gamma(
            'w_scale', alpha=1., beta=0.1, n_samples=n_particles)
    else:
        w_scale = tf.ones([n_particles], dtype=tf.float32)
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        sd = tf.tile(w_scale[:, None, None], [1, n_in+1, n_out])
        if model_type != 'lq':
            sd /= np.sqrt(n_out).astype('f')
        w = bn.normal(
            "w" + str(i), tf.zeros([n_particles, n_in+1, n_out]), std=sd,
            group_ndims=2)
        h = tf.concat([h, tf.ones(tf.shape(h)[:-1])[..., None]], -1) @ w
        if i != len(layer_sizes) - 2:
            h = tf.nn.relu(h)

    if is_regression:
        y_mean = bn.deterministic("y_mean", tf.squeeze(h, -1))
        a0, b0 = 1., 0.1
        y_std = bn.inverse_gamma(
            "y_std", alpha=[a0], beta=[b0], n_samples=n_particles, group_ndims=1)
        assert len(y_std.shape) == 2
        y = bn.normal("y", y_mean, std=y_std)
    else:
        h = tf.nn.log_softmax(h, axis=-1)
        y_mean = bn.deterministic("y_mean", h)
        y = bn.categorical("y", logits=y_mean)

    return bn, None


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
            # observation noise
            std_raw = tf.get_variable(
                'std_raw', shape=[hps.n_particles, 1],
                initializer=tf.constant_initializer(inv_softplus(0.5)))
            svgd_variables['y_std'] = std_raw
            y_std_sym = tf.nn.softplus(std_raw)
            if hps.fix_variance > 0:
                y_std_sym = tf.clip_by_value(y_std_sym, hps.fix_variance, hps.fix_variance + 1e-5)
            svgd_latent['y_std'] = y_std_sym
        # weight variance
        if hps.model_spec == 'lq':
            w_std_raw = tf.get_variable(
                'w_std_raw', shape=[hps.n_particles, ],
                initializer=tf.constant_initializer(inv_softplus(0.5)))
            svgd_variables['w_scale'] = w_std_raw
            w_std_sym = tf.nn.softplus(w_std_raw)
            svgd_latent['w_scale'] = w_std_sym

        meta_model = build_model(
            x, layer_sizes, hps.n_particles, hps.model_spec, hps.regression,
            hps.logits_w_sd)

        def log_joint(bn):
            rv_names = ["w" + str(i) for i in range(len(layer_sizes) - 1)]
            if hps.regression:
                rv_names += ['y_std']
            if hps.model_spec == 'lq':
                rv_names.append('w_scale')
            log_pws = bn.cond_log_prob(rv_names)
            log_py_xw = bn.cond_log_prob('y')
            return tf.add_n(log_pws) + tf.reduce_mean(log_py_xw, 1) * S.n_train

        meta_model.log_joint = log_joint

        # variational: w
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            buf = tf.get_variable(
                'buf_'+str(i),
                initializer=init_bnn_weight(hps.n_particles, n_in, n_out))
            svgd_latent['w'+str(i)] = svgd_variables['w'+str(i)] = buf

        grad_and_var_w, var_bn = stein_variational_gradient_stationary(
            meta_model, {'y': y}, svgd_latent, variables=svgd_variables,
            method=hps.psvi_method)

        optimizer_class = {
            'adam': tf.train.AdamOptimizer,
            'adagrad': tf.train.AdagradOptimizer
        }[hps.optimizer]
        optimizer = optimizer_class(learning_rate=hps.lr)
        global_step = tf.get_variable(
            'global_step', initializer=0, trainable=False)
        infer_op = optimizer.apply_gradients(
            [(-g, v) for g, v in grad_and_var_w], global_step=global_step)

        # prediction: rmse & log likelihood
        log_py_xw = var_bn.cond_log_prob("y")
        log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0))
        if hps.regression:
            y_mean = var_bn["y_mean"]
            y_pred = tf.reduce_mean(y_mean, 0)
            rmse = tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) * S.std_y_train
            log_likelihood -= tf.log(S.std_y_train)
            ystd_avg = tf.reduce_mean(y_std_sym)
        else:
            y_pred = tf.reduce_mean(tf.exp(var_bn['y_mean']), axis=0)
            rmse = 1 - tf.reduce_mean(tf.to_float(tf.nn.in_top_k(y_pred, y, 1)))
            ystd_avg = tf.constant(-1.)

        self.__dict__.update(locals())


def main(hps):
    tf.set_random_seed(hps.seed)
    np.random.seed(hps.seed)
    x_train, y_train, x_valid, y_valid, x_test, y_test, S = load_data(hps)
    M = Model(hps, S)

    """
    data_path = os.path.join(hps.data_dir, hps.dataset + '.data')
    data_func = dataset.data_dict()[hps.dataset]
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_func(data_path)
    if not hps.use_valid:
        x_train = np.vstack([x_train, x_valid])
        y_train = np.hstack([y_train, y_valid])
        x_valid = y_valid = None
    n_train, x_dim = x_train.shape
    x_train, y_train, x_valid, y_valid, x_test, y_test, S = dataset.\
        standardize_new(x_train, y_train, x_valid, y_valid, x_test, y_test, True)
    """

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
            ystd_op=M.ystd_avg, x_ph=M.x, y_ph=M.y, x_extra_ph=None,
            x_valid=x_valid, y_valid=y_valid)

        if len(hps.dump_pred_dir) > 0:
            pred_out = sv.sess_.run([M.var_bn['y_mean'], M.ystd_avg],
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
