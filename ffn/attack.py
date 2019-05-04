import os
import pickle
import sys
import argparse

import tensorflow as tf
from six.moves import range, zip
import numpy as np
import zhusuan as zs
import experiments.utils
import json
from experiments import wrapped_supervisor
from attacks_tf_orig import fgsm, fgsm_targeted

import dataset
import bnn_stein_f, bnn_stein


parser = argparse.ArgumentParser()
parser.add_argument('-ckptd', default='', type=str)
parser.add_argument('-model', default='bnnf', type=str, choices=['bnnf', 'bnn'])
parser.add_argument('-dest', default='/tmp/last.txt', type=str)
args = parser.parse_args()


ckptd = args.ckptd
hps = json.loads(open(ckptd + '/hps.txt').readline())
hps = bnn_stein_f.Object(**hps)

x_train, y_train, x_valid, y_valid, x_test, y_test, S = bnn_stein_f.load_data(hps)

if args.model == 'bnnf':
    M = bnn_stein_f.Model(hps, S)
    log_prob_all = M.var_bn['y_mean_all']
else:
    M = bnn_stein.Model(hps, S)
    log_prob_all = M.var_bn['y_mean']

ckptd = tf.train.get_checkpoint_state(ckptd).model_checkpoint_path
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, ckptd)

# ===================== BUILD GRAPH ======================

probs = tf.reduce_mean(
    tf.nn.softmax(log_prob_all, axis=-1), axis=0)
wrong = M.rmse
stepsize_ph = tf.placeholder(tf.float32, [], name='stepsize')

# clip as done by Y Li
xmin, xmax = x_train.min(), x_train.max()

# while the true input is M.x, we stopped gradient at M.inp
inp_sym = M.inp if args.model == 'bnnf' else M.x
untargeted_adv_inp = fgsm(inp_sym, probs, stepsize_ph, clip_min=xmin, clip_max=xmax)
predent = tf.reduce_mean(tf.reduce_sum(probs * tf.log(probs+1e-7), axis=-1))
mi = predent - \
    tf.reduce_mean(tf.reduce_sum(log_prob_all * tf.exp(log_prob_all), axis=-1))
    
targ_lbl = tf.one_hot(
    tf.zeros([tf.shape(inp_sym)[0]], dtype=tf.int32),
    10)
targeted_adv_inp = fgsm_targeted(
    inp_sym, probs, None, eps=tf.to_float(0.01),
    clip_min=xmin, clip_max=xmax, target_class=0)

# ====================== TARGETED ATTACK ==============================
itr_max = 100
idc = (y_test != 0)
img = x_test[idc]
profs = np.zeros((100, 3))

for i in range(itr_max):
    print('.', end='')
    fd = {
        M.x: img,
        M.y: y_test[idc]
    }
    if args.model == 'bnnf':
        fd[M.x_extra] = x_train[-2:] # anything ok
    adv_img = sess.run(targeted_adv_inp, fd)
    if args.model == 'bnnf':
        adv_img = adv_img[:-2, :]
    img = adv_img
    profs[i, :] = np.array(sess.run([predent, wrong, mi], fd))

np.savetxt(args.dest, profs)

"""
for stepsize in [0.01, 0.02, 0.04, 0.1, 0.5]:
    # idc = (y_test != 0)
    fd = {
        M.x: x_test[idc],
        M.y: y_test[idc],
        M.x_extra: x_train[-10:] # anything ok
    }
    fd[stepsize_ph] = np.cast[np.float32](stepsize)
    adv_img = sess.run(untargeted_adv_inp, fd)[:-10, :]
    fd[M.x] = adv_img
    print(sess.run([predent, wrong, mi], fd))
"""
