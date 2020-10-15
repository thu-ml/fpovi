import tensorflow as tf
import numpy as np
from typing import Optional, Text
import math
import tensorflow_probability as tfp

import sonnet as snt
from sonnet.src import once, utils, initializers, recurrent


def inv_softplus(y): return tf.math.log(tf.math.expm1(y))

def sample_normal(params): return tf.random.normal(params[0].shape, params[0], tf.nn.softplus(params[1]))

def create_normal(mean_init, stddev, name):
    mean = tf.Variable(mean_init, name=f'{name}_mean')
    sd_raw = tf.Variable(tf.zeros_like(mean_init) + inv_softplus(stddev), name=f'{name}_sd_raw')
    prior_sd = tf.Variable(stddev, name=f'{name}_prior_sd', trainable=False)
    return (mean, sd_raw, prior_sd)

def kl_normal(params):
    mean, sd_raw, prior_sd = params
    prior = tfp.distributions.Normal(tf.zeros_like(mean), tf.ones_like(mean) * prior_sd)
    vpost = tfp.distributions.Normal(mean, tf.nn.softplus(sd_raw))
    return tf.reduce_sum(tfp.distributions.kl_divergence(vpost, prior))


class Linear(snt.Linear):
    
    def __init__(self,
                 output_size: int,
                 with_bias: bool = True,
                 w_init: Optional[initializers.Initializer] = None,
                 b_init: Optional[initializers.Initializer] = None,
                 name: Optional[Text] = None,
                 sample_only: bool = False,
                 vi: bool = False
                 ):
        super(Linear, self).__init__(output_size, with_bias, w_init, b_init, name)
        self._sample_only = sample_only
        self._vi = vi

    def kl(self):
        assert self._vi
        return kl_normal(self.w_rv) + (kl_normal(self.b_rv) if self.with_bias else 0)

    @once.once
    def _vi_init(self, inputs: tf.Tensor):
        if self.w_init is None:
            # See https://arxiv.org/abs/1502.03167v3.
            self.w_stddev = 1 / math.sqrt(self.input_size)
            self.w_init = initializers.TruncatedNormal(stddev=self.w_stddev)

        self.w_rv = create_normal(
            self.w_init([self.input_size, self.output_size], inputs.dtype),
            self.w_stddev,
            'w')

        if self.with_bias:
            self.b_rv = create_normal(self.b_init([self.output_size], inputs.dtype), 1., name='b')
    
    def _initialize(self, inputs: tf.Tensor):
        if not self._sample_only and not self._vi:
            return super(Linear, self)._initialize(inputs)  # This is decorated with once

        utils.assert_minimum_rank(inputs, 2)
        input_size = inputs.shape[-1]
        if input_size is None:    # Can happen inside an @tf.function.
            raise ValueError("Input size must be specified at module build time.")

        self.input_size = input_size

        if self._vi:
            self._vi_init(inputs)
            self.w = sample_normal(self.w_rv)
            self.b = sample_normal(self.b_rv)
            return

        if self.w_init is None:
            # See https://arxiv.org/abs/1502.03167v3.
            stddev = 1 / math.sqrt(self.input_size)
            self.w_init = initializers.TruncatedNormal(stddev=stddev)

        self.w = self.w_init([self.input_size, self.output_size], inputs.dtype)

        if self.with_bias:
            self.b = self.b_init([self.output_size], inputs.dtype)

                
class UnrolledLSTM(snt.UnrolledLSTM):

    def __init__(self,
                 hidden_size,
                 w_i_init: Optional[initializers.Initializer] = None,
                 w_h_init: Optional[initializers.Initializer] = None,
                 b_init: Optional[initializers.Initializer] = None,
                 forget_bias: float = 1.0,
                 dtype: tf.DType = tf.float32,
                 name: Optional[Text] = None,
                 sample_only: bool = False,
                 vi: bool = False):
        super(UnrolledLSTM, self).__init__(hidden_size, w_i_init, w_h_init, b_init, forget_bias, dtype, name)
        self._sample_only = sample_only
        self._vi = vi

    def kl(self):
        assert self._vi
        return kl_normal(self._w_i_rv) + kl_normal(self._w_h_rv) + kl_normal(self._b_rv)

    @once.once
    def _vi_init(self, input_sequence):
        utils.assert_rank(input_sequence, 3)    # [num_steps, batch_size, input_size].
        input_size = input_sequence.shape[2]
        dtype = recurrent._check_inputs_dtype(input_sequence, self._dtype)

        w_i_init = self._w_i_init or initializers.TruncatedNormal(
                stddev=1.0 / tf.sqrt(tf.cast(input_size, dtype)))
        w_h_init = self._w_h_init or initializers.TruncatedNormal(
                stddev=1.0 / tf.sqrt(tf.constant(self._hidden_size, dtype=dtype)))
        
        self._w_i_rv = create_normal(
            w_i_init([input_size, 4 * self._hidden_size], dtype),
            1.0 / tf.sqrt(tf.cast(input_size, dtype)),
            'w_i')
        self._w_h_rv = create_normal(
            w_h_init([self._hidden_size, 4 * self._hidden_size], dtype),
            1.0 / tf.sqrt(tf.constant(self._hidden_size, dtype=dtype)),
            'w_h')
        self._b_rv = create_normal(
            self._b_init([4 * self._hidden_size], dtype), 1., 'b')

    def _initialize(self, input_sequence):
        if not self._sample_only and not self._vi:
            return super(UnrolledLSTM, self)._initialize(input_sequence)

        if self._vi:
            self._vi_init(input_sequence)
            self._w_i = sample_normal(self._w_i_rv)
            self._w_h = sample_normal(self._w_h_rv)
            b = sample_normal(self._b_rv)
            b_i, b_f, b_g, b_o = tf.split(b, num_or_size_splits=4)
            b_f += self._forget_bias
            self.b = tf.concat([b_i, b_f, b_g, b_o], axis=0)
            return

        utils.assert_rank(input_sequence, 3)    # [num_steps, batch_size, input_size].
        input_size = input_sequence.shape[2]
        dtype = recurrent._check_inputs_dtype(input_sequence, self._dtype)

        w_i_init = self._w_i_init or initializers.TruncatedNormal(
                stddev=1.0 / tf.sqrt(tf.cast(input_size, dtype)))
        w_h_init = self._w_h_init or initializers.TruncatedNormal(
                stddev=1.0 / tf.sqrt(tf.constant(self._hidden_size, dtype=dtype)))
        
        self._w_i = w_i_init([input_size, 4 * self._hidden_size], dtype)
        self._w_h = w_h_init([self._hidden_size, 4 * self._hidden_size], dtype)

        b_i, b_f, b_g, b_o = tf.split(
                self._b_init([4 * self._hidden_size], dtype), num_or_size_splits=4)
        b_f += self._forget_bias
        self.b = tf.concat([b_i, b_f, b_g, b_o], axis=0)


def lstm_with_recurrent_dropout(hidden_size, dropout=0.5, seed=None, **kwargs):
  if dropout < 0 or dropout >= 1:
    raise ValueError(
        "dropout must be in the range [0, 1), got {}".format(dropout))

  lstm = snt.LSTM(hidden_size, **kwargs)
  rate = snt.LSTMState(hidden=dropout, cell=0.)
  return recurrent._RecurrentDropoutWrapper(lstm, rate, seed), lstm

        
if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    class R():
        def __init__(self, sample_only):
            self.fc = Linear(1, sample_only=sample_only)
        @tf.function
        def __call__(self, inp):
            return self.fc(inp)
        
    print('================ SHOULD BE DIFFERENT ===============')
    r = R(True)
    for _ in range(3):
        print(r(tf.convert_to_tensor([[2., 3.]])))
    try:
        r.fc.variables
    except ValueError:  # empty
        print('ok')
        
    print('================ SHOULD BE THE SAME ===============')
    r = R(False)
    for _ in range(3):
        print(r(tf.convert_to_tensor([[2., 3.]])))
    assert r.fc.variables

    class R():
        def __init__(self, sample_only):
            self.rnn = UnrolledLSTM(1, sample_only=sample_only)
        @tf.function
        def __call__(self, inp):
            return self.rnn(inp, self.rnn.initial_state(tf.shape(inp)[1]))

    print('================ SHOULD BE DIFFERENT ===============')
    r = R(True)
    inp = tf.linspace(-3., 3., 100)[:, None, None]
    seqs = [r(inp)[0][:,0,0] for _ in range(3)]
    print(np.array(seqs)[:, :5])
    try:
        r.rnn.variables
    except ValueError:  # empty
        print('ok')
    r = R(False)
    seqs = [r(inp)[0][:,0,0] for _ in range(3)]
    print('================ SHOULD BE THE SAME ===============')
    print(np.array(seqs)[:, :5])
    assert r.rnn.variables

