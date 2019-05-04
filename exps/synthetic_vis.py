import argparse
import os
import pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='/tmp/dupm')
parser.add_argument('-figs', type=str, default='14')
args = parser.parse_args()
path = args.p


def plot(preds, xs, ys, alpha=0.2, s=None, mode='ci', base=1, plotMean=True, extra_sd=0):
    if not s:
        s = 0.2/preds.shape[0]
    if mode == 'ci':
        mean_ = np.mean(preds, axis=0)
        sd = np.sqrt(np.mean((preds-mean_.reshape((1,-1)))**2, axis=0))
        plt.plot(xs.reshape((-1, )), mean_)
        wsd = 1.96 * (sd**2 + extra_sd**2)**0.5
        plt.fill_between(xs.reshape((-1,)), mean_-wsd, mean_+wsd, facecolor='lightblue',
                         alpha=0.15, interpolate=True)
        plt.fill_between(xs.reshape((-1,)), mean_-1.96*sd, mean_+1.96*sd, facecolor='blue',
                         alpha=0.2, interpolate=True)

    elif mode == 'scatter':
        xs = np.tile(xs.reshape((1, -1)), [preds.shape[0], 1])
        plt.scatter(xs, preds, alpha=alpha, s=s)
    elif mode == 'groundtruth':
        mean_ = np.mean(preds, axis=0)
        sd = np.sqrt(np.mean((preds-mean_.reshape((1,-1)))**2, axis=0))
        plt.plot(xs.reshape((-1, )), mean_-3*sd, linestyle='dashed', c='red', alpha=0.4)
        plt.plot(xs.reshape((-1, )), mean_+3*sd, linestyle='dashed', c='red', alpha=0.4)
    else:
        raise NotImplementedError()

        
def sine_fn(x):
    return x+np.sin(4*x) + np.sin(13*x)


def plot_sine(fil, xlimmax=1.5):
    fil = os.path.join(path, fil)
    plt.xlim(-0.5, xlimmax)
    plt.ylim(-2, 3)
    pred, ylogstd, xs, ys, xtrs, ytrs = pickle.load(open(fil, 'rb'))
    plot(pred, xs, ys, alpha=0.5, mode='ci', extra_sd=0.2**0.5)
    plt.scatter(xtrs, ytrs, s=20, c='red', marker='+')


if args.figs == '14':

    plt.figure(figsize=(10, 5), facecolor='w')
    plt.subplot(231)
    plt.title('w-SGLD')
    plot_sine('wsgld50.bin')
    plt.subplot(234)
    plt.title('f-wSGLD')
    plot_sine('fwsgld50.bin')
    plt.subplot(232)
    plt.title('pi-SGLD')
    plot_sine('pisgld50.bin')
    plt.subplot(235)
    plt.title('f-piSGLD')
    plot_sine('fpisgld50.bin')
    plt.subplot(233)
    plt.title('GFSF')
    plot_sine('gfsf50.bin')
    plt.subplot(236)
    plt.title('f-GFSF')
    plot_sine('fgfsf50.bin')
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(path, 'fig4.png'))
    
    
    plt.figure(figsize=(10, 2), facecolor='w')
    plt.subplot(131)
    plt.title('SVGD')
    plot_sine('svgd50.bin', 2)
    plt.subplot(132)
    plt.title('f-SVGD')
    plot_sine('fsvgd50.bin', 2)
    plt.subplot(133)
    plt.title('HMC')
    plot_sine('hmc.bin', 2)
    plt.savefig(os.path.join(path, 'fig1.png'))

elif args.figs == '5':
    plt.figure(figsize=(12, 6), facecolor='w')
    for i, v in enumerate([5, 10, 50, 100]):
        plt.subplot(2, 4, i+5)
        if i != 0:
            plt.yticks([])
        plt.title('function space, n='+str(v))
        plot_sine('fsvgd{}.bin'.format(v))
        plt.subplot(2, 4, i+1)
        plt.xticks([])
        if i != 0:
            plt.yticks([])
        plt.title('weight space, n='+str(v))
        plot_sine('svgd{}.bin'.format(v))
    plt.savefig(os.path.join(path, 'fig5.png'))

else:
    raise NotImplemented()
