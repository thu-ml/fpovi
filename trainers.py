import tensorflow as tf
import sonnet as snt
import numpy as np

import fpovi
from modules import *
from utils import *


class Trainer(object):

    @staticmethod
    def get_net(args, vdo_rate=None, sample_only=False, vi=False):
        return Net(len(args.pred_targets), args.n_rnns, args.rnn_hid_size, args.fc_hid_size,
                   sample_only=sample_only, vdo_rate=vdo_rate, vi=vi)

    def __init__(self, args, vdo_rate=None, vi=False):
        self.args = args
        self.n_particles = args.n_particles
        self.nets = [Trainer.get_net(args, vdo_rate=vdo_rate, vi=vi) for j in range(args.n_particles)]
        self._y_sd_raw = tf.Variable(-1., dtype=tf.float32)  # 0.3 after softplus
        self.optimizer = tf.optimizers.Adam(learning_rate=args.lr)

    def y_sd(self):
        return tf.nn.softplus(self._y_sd_raw)

    def recon_likelihood(self, pred, y):
        assert len(pred.shape) == len(y.shape)
        return log_normal_pdf(y, pred, self.y_sd())

    def get_loss_and_grad(self):
        raise NotImplementedError()

    @tf.function
    def _train_step(self, x_mb, y_mb):
        loss, grad_and_vars = self.get_loss_and_grad(x_mb, y_mb)
        self.optimizer.apply_gradients(grad_and_vars)
        return loss

    def train_step(self, x_mb, y_mb, dtype=np.float32):
        return self._train_step(tf.constant(x_mb.astype(dtype)),
                                tf.constant(y_mb.astype(dtype))).numpy()
    
    @tf.function
    def _get_loss(self, x_mb, y_mb):
        return self.get_loss_and_grad(x_mb, y_mb)[0]

    def get_loss(self, x_mb, y_mb):
        return self._get_loss(tf.constant(x_mb.astype('f')),
                              tf.constant(y_mb.astype('f'))).numpy()


class SGDTrainer(Trainer):
    """ SGD ensemble training. """
    
    def __init__(self, args):
        super(SGDTrainer, self).__init__(args)

    def get_regularizer(self, net):
        return tf.reduce_sum(
            [tf.reduce_sum(v**2)/2 for v in net.variables]) / self.args.n_observations

    def get_loss_and_grad(self, x_mb, y_mb):
        grad_and_vars = []
        sds = []
        for net in self.nets:
            with tf.GradientTape() as g:
                pred = net(x_mb)
                with g.stop_recording():
                    with tf.GradientTape() as h:
                        h.watch(pred)
                        nll = -self.recon_likelihood(pred, y_mb)
                        nll = tf.reduce_sum(tf.reduce_mean(nll, axis=0))
                        nlj = nll + self.get_regularizer(net)
                        fs_grad, sd_grad = h.gradient(nll * self.args.lh_scale, (pred, self._y_sd_raw))
                        sds.append(sd_grad)
                ws_grads = g.gradient(pred, net.variables, output_gradients=fs_grad)
            grad_and_vars += list(zip(ws_grads, net.variables))
        # optimize y_sd. Not checked
        grad_and_vars += [(tf.reduce_mean(sds), self._y_sd_raw)]
        return nll, grad_and_vars
   
    def predict(self, x_mb):
        outs = np.array([net(x_mb) for net in self.nets])
        return np.mean(outs, axis=0), (np.std(outs, axis=0)**2 + self.y_sd().numpy()**2)**0.5


class VDOTrainer(SGDTrainer):
    """ Recurrent dropout for LSTMs. """

    def __init__(self, args):
        super(SGDTrainer, self).__init__(args, vdo_rate=args.vdo_rate)

    def predict(self, x_mb):
        outs = []
        for _ in range(self.args.vdo_n_samples):
            for net in self.nets:
                outs.append(net(x_mb))
        outs = np.array(outs)
        return np.mean(outs, axis=0), (np.std(outs, axis=0)**2 + self.y_sd().numpy()**2)**0.5


class MFVITrainer(SGDTrainer):

    def __init__(self, args):
        super(SGDTrainer, self).__init__(args, vi=True)

    def get_regularizer(self, net):
        return net.kl() / self.args.n_observations

    def predict(self, x_mb):
        outs = []
        for _ in range(self.args.vdo_n_samples):
            for net in self.nets:
                outs.append(net(x_mb))
        outs = np.array(outs)
        return np.mean(outs, axis=0), (np.std(outs, axis=0)**2 + self.y_sd().numpy()**2)**0.5


class fPOVITrainer(Trainer):
    
    def __init__(self, args):
        super(fPOVITrainer, self).__init__(args)
        self.prior_nets = [Trainer.get_net(args, sample_only=True) for j in range(args.n_particles)]
        
    @property
    def net_variables(self):
        ret = []
        for net in self.nets:
            for w in net.variables:
                ret.append(w)
        return ret

    def get_loss_and_grad(self, x_mb, y_mb):

        args = self.args

        with tf.GradientTape() as g:
            preds = tf.convert_to_tensor([net(x_mb) for net in self.nets])  # [NP, B, ydims]
            prior_preds = tf.convert_to_tensor([net(x_mb) for net in self.prior_nets])

            with g.stop_recording():
                with tf.GradientTape() as h:
                    h.watch(preds)
                    log_lh = self.recon_likelihood(preds, y_mb[None])  # [NP, B, ydims]
                    log_prior = fpovi.get_log_prior(preds, prior_preds, args.n_prior_samples)
                    log_prior = log_prior / args.n_observations  # since log lh is scaled as such
                    log_joint = tf.reduce_sum(tf.reduce_mean(log_lh, axis=1)) + tf.reduce_sum(log_prior)
                    fs_grads, sd_grad = h.gradient(
                        log_joint * args.lh_scale, (preds, self._y_sd_raw))
                new_fs_grads = fpovi.povi(
                    self.n_particles, [fs_grads], [preds], fpovi.rbf_kernel, method=args.fpovi_method)[0][0]

            ws_grads = g.gradient(preds, self.net_variables, output_gradients=new_fs_grads)

        grad_and_vars = [(-g, v) for g, v in zip(ws_grads, self.net_variables)]
        grad_and_vars += [(-sd_grad/self.n_particles, self._y_sd_raw)]
        mean_nll = -log_joint / (1.*self.n_particles)
        # print(mean_nll.numpy())
        return mean_nll, grad_and_vars

    def predict(self, x_mb):
        outs = np.array([net(x_mb) for net in self.nets])
        return np.mean(outs, axis=0), (np.std(outs, axis=0)**2 + self.y_sd().numpy()**2)**0.5
