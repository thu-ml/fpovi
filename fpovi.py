import tensorflow as tf
import tensorflow_probability as tfp


__all__ = ['povi']


def mvn_log_prob(obs, prec, mean):
    dst = obs - mean[None, ...]
    return tf.linalg.diag_part(-1/2 * dst @ prec @ tf.transpose(dst))


def to_double(inp):
    return tf.cast(inp, tf.float64)


def reduce_moments_ax0(t):
    t = to_double(t)
    n = to_double(tf.shape(t)[0])
    mean = tf.reduce_mean(t, axis=0, keepdims=True)
    cov = tf.transpose(t - mean) @ (t - mean) / (n - 1)
    return tf.squeeze(mean, axis=0), cov


def matrix_inverse(m, eig_min, eig_max=1e100): 
    m = to_double(m)
    eig_min = to_double(eig_min)
    uu, vv = tf.linalg.eigh(m)
    uu = 1. / tf.clip_by_value(uu, eig_min, eig_max)
    return vv @ tf.linalg.diag(uu) @ tf.transpose(vv)


def get_log_prior(funcs, prior_funcs, n_prior_samples, jitter=1e-2):
    funcs = tf.reshape(funcs, [tf.shape(funcs)[0], -1])
    prior_funcs = tf.reshape(prior_funcs, [tf.shape(prior_funcs)[0], -1])
    axs = tf.range(tf.shape(funcs)[1])
    idcs = tf.random.shuffle(axs)[:n_prior_samples]
    mean, cov = reduce_moments_ax0(to_double(tf.gather(prior_funcs, idcs, axis=1)))
    prec = matrix_inverse(cov, jitter)
    rt = mvn_log_prob(to_double(tf.gather(funcs, idcs, axis=1)), prec, mean)
    return tf.cast(rt, tf.float32)


def rbf_kernel(theta_x, theta_y, bandwidth='median'):
    """
    :param theta: tensor of shape [n_particles, n_params]
    :return: tensor of shape [n_particles, n_particles]
    """
    n_x = tf.shape(theta_x)[0]
    pairwise_dists = tf.reduce_sum(
        (tf.expand_dims(theta_x, 1) - tf.expand_dims(theta_y, 0)) ** 2,
        axis=-1)
    if bandwidth == 'median':
        bandwidth = tfp.stats.percentile(
            tf.squeeze(pairwise_dists), q=50.)
        bandwidth = 0.5 * bandwidth / tf.math.log(tf.cast(n_x, tf.float32) + 1)
        bandwidth = tf.maximum(tf.stop_gradient(bandwidth), 1e-5)
    Kxy = tf.exp(-pairwise_dists / bandwidth / 2)
    return Kxy, None


def _squeeze(tensors, n_particles):
    return tf.concat(
        [tf.reshape(t, [n_particles, -1]) for t in tensors], axis=1)


def _unsqueeze(squeezed, original_tensors):
    ret = []
    offset = 0
    for t in original_tensors:
        size = tf.reduce_prod(tf.shape(t)[1:])
        buf = squeezed[:, offset: offset+size]
        offset += size
        ret.append(tf.reshape(buf, tf.shape(t)))
    return ret 


def get_n_particles(var_list):
    n_particles = None
    for value_tensor in var_list:
        if n_particles is None:
            n_particles = int(value_tensor.shape[0])
        else:
            assert n_particles == int(value_tensor.shape[0])
    return n_particles


def param_dist(buf, dist_type):
    if dist_type == 'l2':
        k = tf.reduce_sum(
            (buf[None, ...] - buf[:, None, :])**2, axis=-1)
    else:
        k = tf.reduce_max(
            tf.abs(buf[None, ...] - buf[:, None, :]), axis=-1)
    k = tf.reshape(k, [-1])
    return tf.convert_to_tensor(tf.nn.moments(k, axes=[0]))


def povi(n_particles, lh_grads, params, kernel, method='svgd'):
    NP = tf.cast(n_particles, tf.float32)
    params_squeezed = _squeeze(params, n_particles)
    lh_grads = _squeeze(lh_grads, n_particles)
    
    with tf.GradientTape(persistent=True) as g:
        params_sq_1 = tf.identity(params_squeezed)
        g.watch(params_sq_1)
        Kxy, _ = kernel(params_sq_1, params_squeezed)
        _k1 = tf.reduce_sum(Kxy, axis=-1)
        _k2 = Kxy / _k1[None, :]
    # We want dykxy[x] := sum_y\frac{\partial K(x,y)}{\partial y}
    # tf does not support Jacobian, and tf.gradients(Kxy, theta) returns
    # ret[x] = \sum_y\frac{\partial K(x,y)}{\partial x}
    # For stationary kernel ret = -dykxy.
    dykxy = -g.gradient(Kxy, params_sq_1)
    svgd_grads = (tf.matmul(Kxy, lh_grads) + dykxy) / NP
    wsgld_grads = lh_grads - g.gradient(_k2, params_sq_1) + dykxy / _k1[:,None]
    del g

    if method == 'svgd':
        new_grads = svgd_grads
    elif method == 'gfsf':
        new_grads = (lh_grads + tf.linalg.inv(Kxy) @ dykxy) / NP
    elif method == 'wsgld':
        new_grads = wsgld_grads
    elif method == 'pisgld':
        new_grads = (wsgld_grads + svgd_grads) / 2
    else:
        raise NotImplementedError()

    return list(zip(_unsqueeze(new_grads, params), params))
