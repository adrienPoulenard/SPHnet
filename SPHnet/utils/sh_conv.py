import tensorflow as tf
from spherical_harmonics.tf_spherical_harmonics import normalized_sh, unnormalized_sh
import numpy as np

def tf_heaviside(X):
    return tf.maximum(0.0, tf.sign(X))


def tf_segment_indicator_(X, a, b):
    return tf_heaviside(X - a) - tf_heaviside(X - b)


def tf_segment_indictor(X, r, sigma):
    return tf_segment_indicator_(X, r - sigma, r + sigma)


def tf_hat(x, sigma):
    x = x / sigma
    return 0.5*(tf.nn.relu(x + 1.) - 2. * tf.nn.relu(x) + tf.nn.relu(x - 1.))


def tf_gaussian(x, sigma):

    # sigma = 3*sigma
    x2 = tf.multiply(x, x)
    return tf.exp(-x2 / (2. * (sigma ** 2)))


def tf_zero(x, sigma):
    return x


def tf_sh_kernel(X, sq_dist, nr, l_max, sigma, radial_weights_fn, normalize_patch=False, dtype=tf.float32):
    # Y = unnormalized_sh(X, l_max, dtype=dtype)
    Y = normalized_sh(X, l_max, dtype=dtype, eps=0.0001)

    dist = tf.sqrt(tf.maximum(sq_dist, 0.0001))

    if normalize_patch:
        radius = tf.reduce_max(dist, axis=2, keepdims=True)
        dist = tf.divide(dist, radius + 0.0001)

    dist = tf.expand_dims(dist, axis=-1)
    r = (2. * nr * sigma) * tf.reshape(tf.lin_space(start=0., stop=1. - 1. / float(nr), num=nr), shape=(1, 1, 1, nr))
    r = tf.subtract(dist, r)
    radial_weights = radial_weights_fn(r, sigma)
    Y = tf.expand_dims(Y, axis=-1)
    radial_weights = tf.expand_dims(radial_weights, axis=-2)
    y = tf.multiply(Y, radial_weights)

    # y = tf.expand_dims(Y, axis=-1)
    # y = tf.tile(y, multiples=(1, 1, 1, 1, 3))
    return y


def tf_sh_kernel_(X, sq_dist, nr, l_max, rad, radial_weights_fn, normalize_patch=False,
                  radial_first=False,
                  dtype=tf.float32):
    # Y = unnormalized_sh(X, l_max, dtype=dtype)
    Y = normalized_sh(X, l_max, dtype=dtype, eps=0.0001)
    sh = Y

    dist = tf.sqrt(tf.maximum(sq_dist, 0.0001))

    if normalize_patch:
        radius = tf.reduce_max(dist, axis=2, keepdims=True)
        radius = tf.reduce_mean(radius, axis=1, keepdims=True)
        dist = tf.divide(dist, radius + 0.0001)
        rad = 0.75


    dist = tf.expand_dims(dist, axis=-1)
    r = tf.reshape(tf.lin_space(start=0., stop=rad, num=nr), shape=(1, 1, 1, nr))
    r = tf.subtract(dist, r)
    sigma = (rad/(nr - 1))
    radial_weights = radial_weights_fn(r, sigma)
    if radial_first:
        Y = tf.expand_dims(Y, axis=-2)
        radial_weights = tf.expand_dims(radial_weights, axis=-1)
    else:
        Y = tf.expand_dims(Y, axis=-1)
        radial_weights = tf.expand_dims(radial_weights, axis=-2)

    y = tf.multiply(Y, radial_weights)

    # y = tf.expand_dims(Y, axis=-1)
    # y = tf.tile(y, multiples=(1, 1, 1, 1, 3))


    y_w = tf.expand_dims(tf.expand_dims(y[:, :, :, 0, 0], axis=-1), axis=-1)
    y_w = tf.reduce_sum(y_w, axis=2, keepdims=True)
    y = tf.divide(y, y_w + 0.000001)
    return y, sh


def tf_sh_sinusoid_kernel(X, sq_dist, n, l_max, t, sigma, normalize_patch=False, dtype=tf.float32):
    Y = normalized_sh(X, l_max, dtype=dtype, eps=0.0001)

    dist = tf.sqrt(tf.maximum(sq_dist, 0.0001))

    if normalize_patch:
        radius = tf.reduce_max(dist, axis=2, keepdims=True)
        dist = tf.divide(dist, radius + 0.0001)

    t = t / (2*np.pi)

    dist = tf.expand_dims(dist, axis=-1)
    d = t * tf.reshape(tf.lin_space(start=1., stop=n, num=n), shape=(1, 1, 1, n))
    d = tf.multiply(dist, d)
    s = tf.sin(d)
    c = tf.cos(d)

    nb = sq_dist.get_shape()[0]
    nv = sq_dist.get_shape()[1]
    ns = sq_dist.get_shape()[2]
    one = tf.ones(shape=(nb, nv, ns, 1), dtype=tf.float32)

    rw = tf.concat([s, one, c], axis=-1)

    e = tf.expand_dims(tf.exp(-sq_dist/(2*sigma*sigma)), axis=-1)
    rw = tf.multiply(e, rw)
    Y = tf.expand_dims(Y, axis=-1)
    rw = tf.expand_dims(rw, axis=-2)

    y = tf.multiply(Y, rw)
    return y

class ShKernel:
    def __init__(self, nr, l_max, sigma, radial_fn, normalize_patch=False, radial_first=False, return_sh=False):
        self.nr = nr
        self.l_max = l_max
        self.sigma = sigma
        self.radial_fn = radial_fn
        # self.radial_fn = tf_zero
        self.normalize_patch = normalize_patch
        self.radial_first = radial_first
        self.return_sh = return_sh

    def compute(self, X, sq_dist):
        y, sh = tf_sh_kernel_(X, sq_dist,
                            self.nr,
                            self.l_max,
                            self.sigma,
                            self.radial_fn,
                            normalize_patch=self.normalize_patch,
                            radial_first=self.radial_first,
                            dtype=tf.float32)

        if self.return_sh:
            return [y, sh]
        else:
            return y

    def get_shape(self):
        if self.radial_first:
            if self.return_sh:
                return [(self.nr, (self.l_max + 1)**2), ((self.l_max + 1)**2,)]
            else:
                return (self.nr, (self.l_max + 1) ** 2)
        else:
            if self.return_sh:
                return [((self.l_max + 1)**2, self.nr), ((self.l_max + 1)**2,)]
            else:
                return ((self.l_max + 1)**2, self.nr)

def sh_eqvar_conv_1(signal, patches_idx, conv_tensor):
    sampled_signal = tf.gather_nd(signal, patches_idx)
    v = tf.expand_dims(sampled_signal, axis=-2)
    v0 = v[:, :, :, 0, ...]
    v1 = v[:, :, :, 1, ...]
    v2 = v[:, :, :, 2, ...]

    u0 = conv_tensor[:, :, :, 1, ...]
    u1 = conv_tensor[:, :, :, 2, ...]
    u2 = conv_tensor[:, :, :, 3, ...]

    cross_uv0 = tf.multiply(u1, v0) - tf.multiply(u0, v1)
    cross_uv0 = tf.reduce_sum(cross_uv0, axis=2, keepdims=False)

    cross_uv1 = tf.multiply(u2, v0) - tf.multiply(u0, v2)
    cross_uv1 = tf.reduce_sum(cross_uv1, axis=2, keepdims=False)

    cross_uv2 = tf.multiply(u2, v1) - tf.multiply(u1, v2)
    cross_uv2 = tf.reduce_sum(cross_uv2, axis=2, keepdims=False)

    cross_uv = tf.stack([cross_uv0, cross_uv1, cross_uv2], axis=2)

    dot_uv = tf.multiply(u0, v0) + tf.multiply(u1, v1) + tf.multiply(u2, v2)
    dot_uv = tf.reduce_sum(dot_uv, axis=2, keepdims=True)

    return dot_uv, cross_uv


def sh_norm(sh_features, l_max):
    y = sh_features
    L = [y[:, :, 0, ...]]
    p = 1
    for l in range(1, l_max + 1):
        x = y[:, :, p:(p + 2 * l + 1), ...]
        p += 2 * l + 1
        # x = tf.norm(x, axis=2, keepdims=False)
        x = tf.multiply(x, x)
        x = tf.reduce_sum(x, axis=2, keepdims=False)
        x = tf.maximum(x, 0.0001)
        x = tf.sqrt(x)
        L.append(x)
    y = tf.stack(L, axis=2)
    return y

def sh_invar_conv_(signal, patches_idx, conv_tensor, l_max):
    """
    sh_idx = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3], dtype=np.int32)
    sh_idx = sh_idx[:l_max ** 2]

    # sh_idx = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32)
    # sh_idx = sh_idx[:l_max**2-1]
    sh_idx = tf.convert_to_tensor(sh_idx, dtype=tf.int32)
    """


    sampled_signal = tf.gather_nd(signal, patches_idx)
    y = tf.einsum('bvpnr,bvpc->bvnrc', conv_tensor, sampled_signal)
    # y = (1./(float(sampled_signal.get_shape()[2].value)))*y

    L = [y[:, :, 0, ...]]
    p = 1
    for l in range(1, l_max + 1):
        x = y[:, :, p:(p + 2 * l + 1), ...]
        p += 2 * l + 1
        # x = tf.norm(x, axis=2, keepdims=False)
        x = tf.multiply(x, x)
        x = tf.reduce_sum(x, axis=2, keepdims=False)
        x = tf.maximum(x, 0.0001)
        x = tf.sqrt(x)
        L.append(x)
    y = tf.stack(L, axis=2)

    return y


def sh_invar_conv(signal, patches_idx, conv_tensor, kernel, l_max):
    y = sh_invar_conv_(signal, patches_idx, conv_tensor, l_max)
    return tf.einsum('inrj,bvnrj->bvi', kernel, y)

