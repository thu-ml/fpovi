import tensorflow as tf


class Arc:

    @staticmethod
    def sample_basis(n_in, n_rfs):
        return tf.random_normal([n_in, n_rfs], stddev=1.)

    @staticmethod
    def activation(inp, n_rfs):
        return tf.nn.relu(inp) / tf.sqrt(tf.to_float(n_rfs))
