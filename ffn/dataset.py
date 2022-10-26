#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import gzip
import tarfile
import zipfile
import pickle

import numpy as np
import six
from six.moves import urllib, range
from six.moves import cPickle as pickle


import dsdgp_datasets
from utils import Object


def standardize(data_train, data_test):
    """
    Standardize a dataset to have zero mean and unit standard deviation.

    :param data_train: 2-D Numpy array. Training data.
    :param data_test: 2-D Numpy array. Test data.

    :return: (train_set, test_set, mean, std), The standardized dataset and
        their mean and standard deviation before processing.
    """
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    if len(data_train.shape) > 1:
        std *= np.sqrt(data_train.shape[1])
    data_train_standardized = (data_train - mean) / std
    data_test_standardized = (data_test - mean) / std
    mean, std = np.squeeze(mean, 0), np.squeeze(std, 0)
    return data_train_standardized, data_test_standardized, mean, std


def standardize_new(xt, yt, xv, yv, xte, yte, is_regression):
    ss = Object()
    xt, xte, ss.mean_x_train, ss.std_x_train = standardize(xt, xte)
    if xv is not None:
        xv = (xv - ss.mean_x_train[None, :]) / ss.std_x_train[None, :]
    if is_regression:
        yt, yte, ss.mean_y_train, ss.std_y_train = standardize(yt, yte)
        if yv is not None:
            yv = (yv - ss.mean_y_train[None]) / ss.std_y_train[None]
    return xt, yt, xv, yv, xte, yte, ss


def to_one_hot(x, depth):
    """
    Get one-hot representation of a 1-D numpy array of integers.

    :param x: 1-D Numpy array of type int.
    :param depth: A int.

    :return: 2-D Numpy array of type int.
    """
    ret = np.zeros((x.shape[0], depth))
    ret[np.arange(x.shape[0]), x] = 1
    return ret


def download_dataset(url, path):
    print('Downloading data from %s' % url)
    urllib.request.urlretrieve(url, path)


def load_mnist_realval(path, one_hot=False, dequantify=False):
    """
    Loads the real valued MNIST dataset.

    :param path: Path to the dataset file.
    :param one_hot: Whether to use one-hot representation for the labels.
    :param dequantify:  Whether to add uniform noise to dequantify the data
        following (Uria, 2013).

    :return: The MNIST dataset.
    """
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset('http://www.iro.umontreal.ca/~lisa/deep/data/mnist'
                         '/mnist.pkl.gz', path)

    f = gzip.open(path, 'rb')
    if six.PY2:
        train_set, valid_set, test_set = pickle.load(f)
    else:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    x_train, t_train = train_set[0], train_set[1]
    x_valid, t_valid = valid_set[0], valid_set[1]
    x_test, t_test = test_set[0], test_set[1]
    if dequantify:
        x_train += np.random.uniform(0, 1. / 256,
                                     size=x_train.shape).astype('float32')
        x_valid += np.random.uniform(0, 1. / 256,
                                     size=x_valid.shape).astype('float32')
        x_test += np.random.uniform(0, 1. / 256,
                                    size=x_test.shape).astype('float32')
    n_y = t_train.max() + 1
    t_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)
    return x_train, t_transform(t_train), x_valid, t_transform(t_valid), \
        x_test, t_transform(t_test)


def load_binary_mnist_realval(path):
    """
    Loads real valued MNIST dataset for binary classification (Treat 0 & 2-9
    as 0).

    :param path: Path to the dataset file.
    :return: The binary labeled MNIST dataset.
    """
    
    raw = list(load_mnist_realval(path, one_hot=False))
    ret = []
    sizes = []
    for x, t in zip(raw[::2], raw[1::2]):
        maskA = (t == 1) | (t == 7)
        x = x[maskA].astype('f')
        t = (t[maskA] == 1).astype('f')
        sizes.append("%.2f" % ((t > 0).sum() / t.shape[0]))
        ret += [x, t]
    sys.stderr.write("ratio of positive class: {}\n".format(' '.join(sizes)))
    return tuple(ret)


def load_uci_german_credits(path, n_train=400, n_valid=500):
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset('https://archive.ics.uci.edu/ml/'
                         'machine-learning-databases/statlog/'
                         'german/german.data-numeric', path)

    n_dims = 24
    data = np.loadtxt(path)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    x_train = data[:n_train, :n_dims]
    y_train = data[:n_train, n_dims] - 1
    x_valid = data[n_train:n_valid, :n_dims]
    y_valid = data[n_train:n_valid, n_dims] - 1
    x_test = data[n_train:, :n_dims]
    y_test = data[n_train:, n_dims] - 1

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def load_uci_boston_housing(path, dtype=np.float32):
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset('http://archive.ics.uci.edu/ml/'
                         'machine-learning-databases/housing/housing.data',
                         path)

    data = np.loadtxt(path)
    data = data.astype(dtype)
    permutation = np.random.choice(np.arange(data.shape[0]),
                                   data.shape[0], replace=False)
    size_train = int(np.round(data.shape[0] * 0.8))
    size_test = int(np.round(data.shape[0] * 0.9))
    index_train = permutation[0: size_train]
    index_test = permutation[size_train:size_test]
    index_val = permutation[size_test:]

    x_train, y_train = data[index_train, :-1], data[index_train, -1]
    x_val, y_val = data[index_val, :-1], data[index_val, -1]
    x_test, y_test = data[index_test, :-1], data[index_test, -1]

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_uci_protein_data(path, dtype=np.float32):
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset('http://archive.ics.uci.edu/ml/'
                         'machine-learning-databases/00265/CASP.csv',
                         path)

    data = np.loadtxt(open(path), delimiter=',', skiprows=1).astype(dtype)

    permutation = np.random.choice(np.arange(data.shape[0]),
                                   data.shape[0], replace=False)
    size_train = int(np.round(data.shape[0] * 0.8))
    size_test = int(np.round(data.shape[0] * 0.9))
    index_train = permutation[0: size_train]
    index_test = permutation[size_train:size_test]
    index_val = permutation[size_test:]

    x_train, y_train = data[index_train, 1:], data[index_train, 0]
    x_val, y_val = data[index_val, 1:], data[index_val, 0]
    x_test, y_test = data[index_test, 1:], data[index_test, 0]

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_uci_yacht(path, dtype=np.float32):
    uri = 'http://archive.ics.uci.edu/ml/machine-learning-databases/' + \
        '00243/yacht_hydrodynamics.data'

    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset(uri, path)

    data = [l.strip() for l in open(path).readlines() if len(l.strip()) > 0]
    data = [[float(v) for v in l.split(' ') if len(v.strip()) > 0] for l in data]
    data = np.array(data).astype(dtype)

    permutation = np.random.choice(np.arange(data.shape[0]),
                                   data.shape[0], replace=False)
    size_train = int(np.round(data.shape[0] * 0.8))
    size_test = int(np.round(data.shape[0] * 0.9))
    index_train = permutation[0: size_train]
    index_test = permutation[size_train:size_test]
    index_val = permutation[size_test:]

    x_train, y_train = data[index_train, :-1], data[index_train, -1]
    x_val, y_val = data[index_val, :-1], data[index_val, -1]
    x_test, y_test = data[index_test, :-1], data[index_test, -1]

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_cifar10(path, normalize=True, dequantify=False, one_hot=True):
    """
    Loads the cifar10 dataset.
    :param path: Path to the dataset file.
    :param normalize: Whether to normalize the x data to the range [0, 1].
    :param dequantify: Whether to add uniform noise to dequantify the data
        following (Uria, 2013).
    :param one_hot: Whether to use one-hot representation for the labels.
    :return: The cifar10 dataset.
    """
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset(
            'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', path)

    data_dir = os.path.dirname(path)
    batch_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    if not os.path.isfile(os.path.join(batch_dir, 'data_batch_5')):
        with tarfile.open(path) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, data_dir)

    train_x, train_y = [], []
    for i in range(1, 6):
        batch_file = os.path.join(batch_dir, 'data_batch_' + str(i))
        with open(batch_file, 'rb') as f:
            if six.PY2:
                data = pickle.load(f)
            else:
                data = pickle.load(f, encoding='latin1')
            train_x.append(data['data'])
            train_y.append(data['labels'])
    train_x = np.vstack(train_x)
    train_y = np.hstack(train_y)

    test_batch_file = os.path.join(batch_dir, 'test_batch')
    with open(test_batch_file, 'rb') as f:
        if six.PY2:
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='latin1')
        test_x = data['data']
        test_y = np.asarray(data['labels'])

    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    if dequantify:
        train_x += np.random.uniform(0, 1,
                                     size=train_x.shape).astype('float32')
        test_x += np.random.uniform(0, 1, size=test_x.shape).astype('float32')
    if normalize:
        train_x = train_x / 256
        test_x = test_x / 256

    train_x = train_x.reshape((50000, 3, 32, 32)).transpose(0, 2, 3, 1)
    test_x = test_x.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
    t_transform = (lambda x: to_one_hot(x, 10)) if one_hot else (lambda x: x)
    return train_x, t_transform(train_y), test_x, t_transform(test_y)


def load_cifar10_bin(path):
    train_x, train_y, test_x, test_y = load_cifar10(path)
    def transform(x, y):
        x = x.reshape((x.shape[0], -1))
        m = (y[:, 0] > 0) | (y[:, 2] > 0)
        return x[m].astype('f'), (y[m, 0] > 0).astype('f')
    val_pr = 40000
    x_train, y_train = transform(train_x, train_y)
    x_train = x_train[:val_pr]
    y_train = y_train[:val_pr]
    x_val = x_train[val_pr:]
    y_val = y_train[val_pr:]
    x_test, y_test = transform(test_x, test_y)
    return x_train, y_train, x_val, y_val, x_test, y_test


def load_year(path):
    """
    train: first 463,715 examples 
    test: last 51,630 examples 
    """
    raw_path = os.path.join(os.path.dirname(path), 'YearPredictionMSD.txt')
    assert os.path.isfile(raw_path) # TODO
    if not os.path.isfile(path):
        x_and_y = np.loadtxt(raw_path, delimiter=',')
        with open(path, 'wb') as fout:
            pickle.dump(x_and_y, fout)
    else:
        with open(path, 'rb') as fin:
            x_and_y = pickle.load(fin)
    y = x_and_y[:, 0].astype('f')
    x = x_and_y[:, 1:].astype('f')
    # train_test_split = 463715
    # x_train0, y_train0 = x[:train_test_split], y[:train_test_split]
    # x_test, y_test = x[train_test_split:], y[train_test_split:]
    indices = np.arange(x_and_y.shape[0])
    np.random.shuffle(indices)
    train_test_split = int(x_and_y.shape[0] * 0.9)
    x_train0, y_train0 = x[indices[:train_test_split]], \
        y[indices[:train_test_split]]
    x_test, y_test = x[indices[train_test_split:]], \
        y[indices[train_test_split:]]
    # 
    train_valid_split = int(train_test_split * 0.9)
    indices = np.arange(x_train0.shape[0])
    np.random.shuffle(indices)
    x_train, y_train = x_train0[indices[:train_valid_split]], \
        y_train0[indices[:train_valid_split]]
    x_valid, y_valid = x_train0[indices[train_valid_split:]], \
        y_train0[indices[train_valid_split:]]
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def load_cubic100(path, n_train=100):
    fn = lambda x: ((x**3) + np.random.normal(size=x.shape) * 3).squeeze()
    x_train = np.random.uniform(-4, 4, size=n_train // 2).reshape((-1, 1))
    x_valid = np.random.uniform(-4, 4, size=n_train // 2).reshape((-1, 1))
    x_test = np.linspace(-6, 6, 1000).reshape((-1, 1))
    return tuple(
        map(lambda a: a.astype('f'),
            [x_train, fn(x_train), x_valid, fn(x_valid), x_test, fn(x_test)]))


def load_cubic10(path, n_train=10):
    fn = lambda x: ((x**3) + np.random.normal(size=x.shape) * 3).squeeze()
    x_train = np.random.uniform(-4, 4, size=n_train // 2).reshape((-1, 1))
    x_valid = np.random.uniform(-4, 4, size=n_train // 2).reshape((-1, 1))
    x_test = np.linspace(-6, 6, 1000).reshape((-1, 1))
    return tuple(
        map(lambda a: a.astype('f'),
            [x_train, fn(x_train), x_valid, fn(x_valid), x_test, fn(x_test)]))


def load_sine(path):
    np.random.seed(233)
    def fn(x):
        eps = np.random.normal(size=x.shape) * 0.03
        y = x + eps + np.sin(4*(x+eps)) + np.sin(13*(x+eps))
        return y.squeeze()
    # x_train and x_valid will be concatenated
    x_train = np.random.uniform(0, 0.6, size=12).reshape((-1, 1))
    x_valid = np.random.uniform(0.8, 1, size=8).reshape((-1, 1))
    x_test = np.linspace(-1, 2, 1000).reshape((-1, 1))
    return tuple(
        map(lambda a: a.astype('f'),
            [x_train, fn(x_train), x_valid, fn(x_valid), x_test, fn(x_test)]))


def wrap_dsdgp_datasets(Class):
    def load(path):
        c = Class()
        c.data_path = os.path.dirname(path) + '/'
        ret = c.get_data(prop=0.9)
        ret['Y'] = ret['Y'].squeeze()
        ret['Ys'] = ret['Ys'].squeeze()
        spl = ret['X'].shape[0] * 4 // 5
        return ret['X'][:spl], ret['Y'][:spl], ret['X'][spl:], ret['Y'][spl:], \
            ret['Xs'], ret['Ys']
    return load


def data_dict():
    import inspect
    funcs = [(name,obj) for name,obj in inspect.getmembers(sys.modules[__name__]) 
             if inspect.isfunction(obj) and name.startswith('load_uci_')]
    ret = dict((name.replace('load_uci_', ''), obj) for name, obj in funcs)

    for k in ['Concrete', 'Energy', 'Kin8mn', 'Naval', 'Power', 'WineRed', 'WineWhite']:
        ret[k.lower()] = wrap_dsdgp_datasets(dsdgp_datasets.__dict__[k])

    ret['year'] = load_year
    ret['mnist'] = load_mnist_realval
    ret['cifar10'] = load_cifar10_bin
    ret['cubic100'] = load_cubic100
    ret['cubic10'] = load_cubic10
    ret['sine'] = load_sine
    return ret
