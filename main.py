import tensorflow as tf
import sonnet as snt
import numpy as np
import tqdm
from experiments.slave import *
import experiments.utils
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os.path

import data
import trainers


parser = experiments.utils.parser('frnn')
parser.add_argument('-n_rnns', type=int, default=1, help='depth of the RNN model')
parser.add_argument('-rnn_hid_size', type=int, default=40)
parser.add_argument('-fc_hid_size', type=int, default=100)
parser.add_argument('-seq_len', type=int, default=48)
parser.add_argument('-n_particles', type=int, default=8)
parser.add_argument('-pred_targets', type=(lambda s:[int(t) for t in s.split(',')]),
                    default=[1, 13, 25], help='lags to predict')
parser.add_argument('-pred_type', type=int, default=0)
parser.add_argument('-lr', type=float, default=4e-3)
parser.add_argument('-n_epochs', type=int, default=60)
parser.add_argument('-batch_size', type=int, default=100)
parser.add_argument('-lh_scale', type=float, default=1.,
                    help="likelihood scale, shouldn't need tuning")
parser.add_argument('-trainer', type=str, default='fpovi')
parser.add_argument('-vdo_rate', type=float, default=0.5)
parser.add_argument('-vdo_n_samples', type=int, default=8)
parser.add_argument('-fpovi_method', type=str, default='pisgld')
parser.add_argument('-n_prior_samples', type=int, default=4)
parser.add_argument('-data_path', type=str,
                    default=os.path.expanduser('~/data/gef/temp-load.npz'))


def load_data(args):
    ddict = np.load(args.data_path)
    X, Yall, Y = ddict['X'], ddict['Yall'], ddict['Y']
    spl = -365
    Xtr, Ytr, ds_stats = data.process_data(
        X[:spl], Yall[:, :spl], Y[:spl], args.seq_len, args.pred_targets, args.pred_type)
    Xte, Yte, _        = data.process_data(
        X[spl:], Yall[:, spl:], Y[spl:], args.seq_len, args.pred_targets, args.pred_type)
    Xtr, Ytr = data.normalize_dset(Xtr, Ytr, ds_stats)
    Xte, Yte = data.normalize_dset(Xte, Yte, ds_stats)
    val_spl = -200
    Xtr, Xva = Xtr[:, :val_spl], Xtr[:, val_spl:]
    Ytr, Yva = Ytr[:val_spl], Ytr[val_spl:]
    return Xtr, Ytr, Xte, Yte, Xva, Yva, ds_stats


def main(args):

    Xtr, Ytr, Xte, Yte, Xva, Yva, ds_stats = load_data(args)
    args.n_observations = Xtr.shape[1]
    trainer = {
        'sgd': trainers.SGDTrainer,
        'vdo': trainers.VDOTrainer,
        'fpovi': trainers.fPOVITrainer,
        'mfvi': trainers.MFVITrainer,
    }[args.trainer](args)
    
    with tqdm.trange(args.n_epochs) as trg:
        for c_ep in trg:
            idcs = np.arange(Xtr.shape[1])
            np.random.shuffle(idcs)
            Xtr, Ytr = Xtr[:,idcs], Ytr[idcs]
            losses = []
            for b in range(0, Xtr.shape[1], args.batch_size):
                l = trainer.train_step(Xtr[:,b:b+args.batch_size], Ytr[b:b+args.batch_size])
                losses.append(l)
            val_loss = trainer.get_loss(Xva, Yva)
            to_print = {
                'train_loss': np.mean(losses),
                'val_loss': val_loss
            }
            if c_ep % 5 == 0:
                val_pred, _ = trainer.predict(Xva)
                to_print['val_rmse'] = ((val_pred - Yva) ** 2).mean() ** 0.5
            trg.set_postfix(**to_print)

    test_pred_mean, test_pred_sd = trainer.predict(Xte)
    rmse = ((test_pred_mean - Yte) ** 2).mean() ** 0.5
    loglh = -(test_pred_mean-Yte)**2 / (2*test_pred_sd**2) - np.log(test_pred_sd * np.sqrt(np.pi * 2))

    lo = test_pred_mean - 1.96*test_pred_sd
    hi = test_pred_mean + 1.96*test_pred_sd
    avg = np.logical_and(Yte>=lo, Yte<=hi).astype('f').mean()
    print(f'Test rmse = {rmse}, nll = {-loglh.mean()}, CI cvg = {avg}')

    P = len(args.pred_targets)
    with Canvas(figsize=(10, 3*P), nrow=P) as cv:
        S = 400
        X = np.arange(test_pred_mean[-S:].shape[0])
        for p in range(P):
            cv[p].plot(X, test_pred_mean[-S:][:,p], label='pred')
            cv[p].plot(X, Yte[-S:][:,p], label='act')
            cv[p].fill_between(
                X,
                test_pred_mean[-S:][:,p] - 2*test_pred_sd[-S:][:,p],
                test_pred_mean[-S:][:,p] + 2*test_pred_sd[-S:][:,p],
                alpha=0.2)
            cv[p].legend()
        plt.imsave(os.path.join(args.dir, 'pred.png'), cv.dump())


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    args = parser.parse_args()
    experiments.utils.preflight(args)
    np.random.seed(1234)
    main(args)
