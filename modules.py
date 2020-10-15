""" adapted from sonnet.modules for various inference methods """
import tensorflow as tf
import sonnet as snt
import numpy as np

import layers
import fpovi


class Attn(snt.Module):
    
    def __init__(self, hid_size, sample_only=False, vi=False):
        super(Attn, self).__init__()
        self.W1 = layers.Linear(hid_size, sample_only=sample_only, vi=vi)
        self.W2 = layers.Linear(hid_size, sample_only=sample_only, vi=vi)
        self.V = layers.Linear(1, sample_only=sample_only, vi=vi)
        
    def __call__(self, query, values):
        """
        :param query: [B, inp_hid_dims]
        :param values: [T, B, inp_hid_dims]
        """
        def c(fc, inp):
            assert len(inp.shape) == 3
            out_shape = tf.convert_to_tensor(
                [tf.shape(inp)[0], tf.shape(inp)[1], fc.output_size])
            inp = tf.reshape(inp, [-1, inp.shape[2]])
            return tf.reshape(fc(inp), out_shape)
        score = c(self.V,
                  tf.nn.tanh(c(self.W1, values) + c(self.W2, query[None])))  # [T,B,1]
        attn_weights = tf.nn.softmax(score, axis=0)  # [T,B,1]
        cvec = tf.reduce_sum(attn_weights * values, axis=0)
        return cvec  # [B,inp_hid_dims]

    def kl(self):
        return self.W1.kl() + self.W2.kl() + self.V.kl()


class Net(snt.Module):

    def __init__(self, n_preds, n_rnns=1, rnn_hid_size=40, hid_size=100,
                 sample_only=False, vi=False, vdo_rate=None):
        assert not (sample_only and vdo_rate), NotImplementedError()
        super(Net, self).__init__()
        self.n_rnns, self.rnn_hid_size, self.n_preds = n_rnns, rnn_hid_size, n_preds
        self.vdo_rate = vdo_rate
        if vdo_rate is None:
            self.rnns = [layers.UnrolledLSTM(rnn_hid_size, sample_only=sample_only, vi=vi)
                         for _ in range(n_rnns)]
        else:
            self.rnns = [
                layers.lstm_with_recurrent_dropout(rnn_hid_size, dropout=vdo_rate, seed=1234)
                for _ in range(n_rnns)]
        self.attns, self.fcs = [], []
        for k in range(n_preds):
            self.attns.append(Attn(rnn_hid_size+10, sample_only=sample_only, vi=vi))
            fc1 = layers.Linear(hid_size, sample_only=sample_only, vi=vi)
            fc2 = layers.Linear(1, sample_only=sample_only, vi=vi)
            self.fcs.append((fc1, fc2))

    def kl(self):
        ret = tf.add_n([rnn.kl() for rnn in self.rnns])
        ret += tf.add_n([a.kl() for a in self.attns])
        ret += tf.add_n([a.kl()+b.kl() for a,b in self.fcs])
        return ret

    def __call__(self, inp):
        """
        :param inp: [T, B, xdims]
        :return: [B, n_preds]
        """
        cur = inp
        if self.vdo_rate is None:
            for rnn in self.rnns:
                init_st = rnn.initial_state(inp.shape[1])
                cur, last_state = rnn(cur, init_st)
        else:
            for train_core, _ in self.rnns:
                # always use train core (w/dropout) since we need uncertainty in test
                init_st = train_core.initial_state(inp.shape[1])
                cur, last_state = snt.dynamic_unroll(train_core, cur, init_st)
            last_state = last_state[0]  # 1 is dropout mask
        out_seq = cur
        last_state = last_state.hidden
        assert out_seq.shape.as_list()[-1] == self.rnn_hid_size

        outs = []
        for k in range(self.n_preds):
            attn_out = self.attns[k](tf.nn.relu(last_state), tf.nn.relu(out_seq))
            fc1, fc2 = self.fcs[k]
            hid = tf.nn.relu(fc1(attn_out))
            if self.vdo_rate is not None:
                hid = tf.nn.dropout(hid, self.vdo_rate)
            out = fc2(hid)
            assert len(out.shape) == 2 and out.shape[-1] == 1, out.shape
            outs.append(out)
        return tf.concat(outs, axis=-1)

