import tensorflow as tf
import numpy as np


def log_normal_pdf(x, mean, sd):
    return -(((x-mean)/sd)**2 / 2) - tf.math.log(sd) -\
        tf.cast(tf.math.log(np.pi * 2) / 2, tf.float32)
