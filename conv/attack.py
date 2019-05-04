import os, sys, re

import tensorflow as tf
import numpy as np
import tensorpack
from cleverhans.attacks_tf import fgm

from bayes_convnet import Model, parser, get_data
from tqdm import tqdm


def entropy_of(lpb):
    prb = np.exp(lpb).mean(axis=0)
    return -(prb * np.log(prb)).sum(axis=-1).mean()


ckpt_dir = sys.argv[1]
if len(sys.argv) > 2:
    single_model = (sys.argv[2] == 'single')
    print("S:", single_model)
else:
    single_model = False

# build args
args = open(os.path.join(ckpt_dir, 'log.log')).readline().split('bayes_convnet.py ')[1].rstrip().split(' ')
args = parser.parse_args(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
from tensorflow.python.client import device_lib
devices = [d for d in device_lib.list_local_devices() if d.device_type=='GPU']
setattr(args, 'devices', devices)
setattr(args, 'n_train', 90000) # Should be useless, just a placeholder

# build graph & restore from ckpt
with tensorpack.TowerContext('', is_training=False):
    model = Model(args)
    image_ph, label_ph = model.inputs()
    model.build_graph(image_ph, label_ph)

cfg = tf.ConfigProto(allow_soft_placement=True)
sess = tf.InteractiveSession(config=cfg)
ckpt_dir = tf.train.get_checkpoint_state(ckpt_dir).model_checkpoint_path
saver = tf.train.Saver()
saver.restore(sess, ckpt_dir)

if args.variational == 'f_svgd':
    log_prob = tf.get_default_graph().get_tensor_by_name('variational/all_func:0')
#   log_prob = tf.get_default_graph().get_tensor_by_name('variational/concat_1:0')
    correct = tf.get_default_graph().get_tensor_by_name('correct:0')
else:
    log_prob = tf.get_default_graph().get_tensor_by_name('variational/concat:0')
    correct = tf.get_default_graph().get_tensor_by_name('correct:0')

assert len(log_prob.get_shape().as_list()) == 3
if single_model:
    probs = tf.nn.softmax(log_prob[0])
    correct = tf.reduce_mean(tf.to_float(tf.nn.in_top_k(probs, label_ph, 1)))
else:
    log_probs = tf.log(tf.reduce_mean(tf.nn.softmax(log_prob), axis=0))
    probs = tf.nn.softmax(log_probs)
    # correct = tf.reduce_mean(tf.to_float(tf.nn.in_top_k(probs, label_ph, 1)))
    
ds_train, _ = get_data('train', args)
ds_test, _ = get_data('test', args)

# Untargeted BIM
from tensorpack.dataflow import dataset
pp_mean = dataset.Cifar10('train').get_per_pixel_mean()

stepsize_ph = tf.placeholder(tf.float32, [])
orig_input_ph = tf.placeholder(tf.float32, image_ph.get_shape().as_list())
# adv_inp = fgm(image_ph, probs, y=tf.one_hot(label_ph, depth=10), eps=tf.to_float(1))
pp_mean_sym = tf.tile(tf.constant(pp_mean[None]), [tf.shape(image_ph)[0], 1, 1, 1])
# adv_inp = tf.clip_by_value(adv_inp, -pp_mean_sym, 255 - pp_mean_sym)
# adv_inp = tf.clip_by_value(adv_inp, orig_input_ph - stepsize_ph, orig_input_ph + stepsize_ph)
adv_inp = fgm(image_ph, probs, #y=tf.one_hot(label_ph, depth=10)
              eps=tf.to_float(stepsize_ph))
adv_inp = tf.clip_by_value(adv_inp, -pp_mean_sym, 255 - pp_mean_sym)
adv_inp = tf.clip_by_value(adv_inp, orig_input_ph - stepsize_ph, orig_input_ph + stepsize_ph)

for EPSILON in [0, 1, 2, 4, 8, 16]:
    ds_test.reset_state()
    crts, ents = [], []
    with tqdm(total=10000) as pbar:
        for i, (img, lbl) in tqdm(enumerate(ds_test.get_data())):
            if img.shape[0] != 128:
                break
            fd = {
                image_ph: img,
                label_ph: lbl,
                orig_input_ph: img,
                stepsize_ph: np.cast[np.float32](EPSILON)
            }
            adv_img = sess.run(adv_inp, fd)
            fd[image_ph] = adv_img
            lpb, crt = sess.run([log_prob, correct], fd)
            ent = entropy_of(lpb)
            crts.append(crt)
            ents.append(ent)
            pbar.update(128)
    print('\n\n', EPSILON, np.mean(crts), np.mean(ents), '\n')
