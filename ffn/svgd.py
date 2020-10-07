"""
Code baesd on Qiang Liu's original repo.
"""
import tensorflow as tf
import zhusuan as zs
import sys


__all__ = ['stein_variational_gradient']


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
        bandwidth = tf.contrib.distributions.percentile(
            tf.squeeze(pairwise_dists), q=50.)
        bandwidth = 0.5 * bandwidth / tf.log(tf.cast(n_x, tf.float32) + 1)
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


def _svgd_stationary(n_particles, log_joint, params, kernel,
                     replace_grad=None, additional_grad=None, profile=False,
                     method='svgd'):
    """
    POVI using a stationary kernel.
    :param log_joint: tensor representing the log joint density. Alternatively,
                      you can provide its gradient through the `replace_grad`
                      and `additional_grad` arguments.
    :param params: model parameters
    :param method: POVI method to use
    """
    params_squeezed = _squeeze(params, n_particles)
    Kxy, dykxy = kernel(params_squeezed, tf.stop_gradient(params_squeezed))

    # We want dykxy[x] := sum_y\frac{\partial K(x,y)}{\partial y}
    # tf does not support Jacobian, and tf.gradients(Kxy, theta) returns 
    # ret[x] = \sum_y\frac{\partial K(x,y)}{\partial x}
    # For stationary kernel we have dykxy = -ret. 
    if dykxy is None:
        dykxy = -tf.gradients(Kxy, params_squeezed)[0]
    else:
        assert False # required by current impl of wsgld. also is it deprecated?
        dykxy = dykxy(Kxy, params_squeezed)

    if replace_grad is None:
        grads = tf.gradients(log_joint, params)
    else:
        grads = replace_grad
    if additional_grad is not None:
        grads = [g1 + g2 for g1, g2 in zip(grads, additional_grad)]
    grads = _squeeze(grads, n_particles)

    svgd_grads = (tf.matmul(Kxy, grads) + dykxy) / tf.to_float(n_particles)
    _k1 = tf.stop_gradient(tf.reduce_sum(Kxy, axis=-1))
    wsgld_grads = grads + \
        -tf.gradients(Kxy / _k1[None, :], params_squeezed)[0] + \
        dykxy / _k1[:, None]

    if method == 'svgd':
        new_grads = svgd_grads
    elif method == 'gfsf':
        new_grads = (grads + tf.matrix_inverse(Kxy) @ dykxy) / tf.cast(
            n_particles, tf.float32)
    elif method == 'wsgld':
        new_grads = wsgld_grads
    elif method == 'pisgld':
        new_grads = (wsgld_grads + svgd_grads) / 2
    else:
        raise NotImplementedError()

    ret = list(zip(_unsqueeze(new_grads, params), params))

    if not profile:
        return ret

    assert method == 'svgd', 'other things not implemented'
    lh_grad_mixed = tf.matmul(Kxy, grads)
    lh_grad_mixed_l2 = tf.reduce_sum(lh_grad_mixed ** 2, axis=-1)
    orig_grad_l2 = tf.reduce_sum(grads ** 2, axis=-1)
    innerp = tf.reduce_sum(grads * lh_grad_mixed, axis=-1) / tf.sqrt(
        lh_grad_mixed_l2 * orig_grad_l2 + 1e-5)
    lInf = lambda a: tf.reduce_max(tf.abs(a), -1)

    prof = {
        'avg_grad_l2': tf.reduce_mean(lh_grad_mixed_l2),
        'avg_grad_li': tf.reduce_mean(lInf(lh_grad_mixed)),
        'avg_org_grad_l2': tf.reduce_mean(orig_grad_l2),
        'avg_org_grad_li': tf.reduce_mean(lInf(grads)),
        'avg_grad_innerp': tf.reduce_mean(innerp),
        'avg_repulsive_l2': tf.reduce_mean(tf.reduce_sum(dykxy**2,axis=-1)),
        'avg_repulsive_li': tf.reduce_mean(lInf(dykxy)),
        'param_l2': param_dist(params_squeezed, 'l2'),
        'param_linf': param_dist(params_squeezed, 'linf'),
    }
    prof_k = list(prof)
    prof_v = [prof[k] for k in prof_k]
    return ret, (prof_k, prof_v)


def stein_variational_gradient_stationary(
    forward_model, observed, latent, variables=None, kernel=None,
    method='svgd', profile=False):
    """
    :param forward_model: meta_bn whose `observe` method returns
                          the model BN and optionally other things for kernel
    :param observed: same as v3
    :param latent: dict((name, value_buf)) where value_buf is variable of shape
                   [n_particles, ...]
    :param dykxy: f : R^{m*m} * R^{m*d} -> R^m, s.t.
                  f(Kxy, X)_x = \\sum_y \\frac{\\partial K(x,y)}{\\partial y}
    """
    kernel = kernel or rbf_kernel

    if variables is None:
        variables = latent

    var_list = [v for _, v in variables.items()]
    n_particles = get_n_particles(var_list)
    observed = observed.copy()
    observed.update(latent)
    bn, _ = forward_model.observe(**observed)
    log_joint = bn.log_joint()

    grad_and_vars = _svgd_stationary(
        n_particles, log_joint, var_list, kernel, method=method, profile=profile)
    return grad_and_vars, bn



def svgd_act_kernel(n_particles, log_joint, all_activations, params,
                    kernel_type):
    if kernel_type == 'cosine' or kernel_type.find('norm') != -1:
        sys.stderr.write("Using normalized activation\n")
        for i, a in enumerate(all_activations):
            assert len(a.shape) == 3 and int(a.shape[0]) == n_particles
            if kernel_type.startswith('laplace'):
                a = a / (1e-5 + tf.reduce_sum(
                    tf.abs(a), axis=[0, 1], keepdims=True))
            else:
                a = a / tf.sqrt(
                    1e-5 + tf.reduce_sum(a**2, axis=[0, 1], keepdims=True))
            all_activations[i] = tf.reshape(a, [n_particles, -1])

    all_activations = tf.concat(all_activations, axis=1)

    def Kfxfy_and_grad(fx, fy):
        if kernel_type == 'cosine':
            K = fx @ tf.transpose(fy)
            dyKxy_xth = lambda x: tf.tile(fx[x:x+1, :], [n_particles, 1])
        elif kernel_type.startswith('laplace'):
            fxt = fx[:, None, :]
            fyt = fy[None, ...]
            K = tf.exp(-tf.reduce_sum(tf.abs(fxt - fyt), axis=-1))
            dyKxy_xth = lambda x: K[x, :, None] * \
                tf.sign(tf.abs(fx[x:x+1, :] - fy))
        return K, dyKxy_xth

    K, dyKxy_xth = Kfxfy_and_grad(all_activations, all_activations)
    #    grad dykxy[x] 
    # := sum_y\frac{\partial K(f(x), f(y))}{\partial y}
    #  = sum_y\frac{\partial K(f(x), f(y))}{\partial f(y)} \nabla f(y)
    #  |------------- grad_ys ---------------------------|
    grad_lists = [[] for _ in params]
    for x in range(n_particles):
        grad_ys = dyKxy_xth(x)
        grad_xs = tf.gradients(all_activations, params, grad_ys=grad_ys)
        for i, g in enumerate(grad_xs):
            if g is None:
                if x == 0:
                    sys.stderr.write("WARNING: part {} in dyKxy is zero\n".\
                                     format(params[i].name))
                g = tf.zeros_like(params[i])
            grad_lists[i].append(tf.reduce_sum(g, axis=0, keepdims=True))
    grad_y_Kxys = [tf.concat(gl, axis=0) for gl in grad_lists]

    grad_ll = tf.gradients(log_joint, params)
    grads = [(g1 + g2) / tf.cast(n_particles, tf.float32)
             for g1, g2 in zip(grad_y_Kxys, grad_ll)]

    return list(zip(grads, params))


def stein_variational_gradient_act_kernel(
        forward_model, observed, latent, variables=None, kernel_type=None):
    """
    :param forward_model: meta_bn whose `observe` method returns
                          the model BN and optionally other things for kernel
    :param observed: same as v3
    :param latent: dict((name, value_buf)) where value_buf is variable of shape
                   [n_particles, ...]
    """
    if variables is None:
        variables = latent
    var_list = [v for _, v in variables.items()]
    n_particles = get_n_particles(var_list)

    observed = observed.copy()
    observed.update(latent)
    bn, all_activations = forward_model.observe(**observed)
    log_joint = bn.log_joint()

    grad_and_vars = svgd_act_kernel(
        n_particles, log_joint, all_activations, var_list, kernel_type)
    return grad_and_vars, bn

