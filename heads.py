import numpy as np
import tensorflow as tf
import math
from controller import Controller


class Head(object):

    def __init__(self, memory_capacity, memory_vector_size):

        self._memory_capacity = memory_capacity
        self._memory_vector_size = memory_vector_size

        self._controller = Controller(self._memory_capacity, self._memory_vector_size)

        # addressing emissions
        self._k_t = None
        self._beta_t = None
        self._g_t = None
        self._s_t = None
        self._gamma_t = None

        # addresses
        self._w_last = tf.Variable(initial_value=tf.ones(shape=(self._memory_capacity,),
                                                         dtype=tf.float32,
                                                         name='w_last') / self._memory_capacity,
                                   trainable=False)

    def produce_address(self, X, M):

        # emit controller parameters for addressing
        self._k_t, self._beta_t, self._g_t, self._s_t, self._gamma_t = self._controller.emit_heads_parameters(X)

        with tf.variable_scope('addressing_mechanism'):
            # flow of addressing mechanism
            wc_t = self._content_addressing(M)
            wg_t = self._interpolation(wc_t)
            w_tilde_t = self._convolutional_shift(wg_t)
            w_t = self._sharpening(w_tilde_t)

        return w_t

    def _content_addressing(self, M):

        assert self._k_t is not None and self._beta_t is not None, 'Must call `emit_head_parameters` first.'

        with tf.variable_scope('content_addressing'):
            # compute norm of key and memory content
            k_t_norm = 1#tf.sqrt(tf.reduce_sum(tf.pow(self._k_t, 2)))
            M_norm = tf.sqrt(tf.reduce_sum(tf.pow(M, 2), axis=1))

            # compute cosine similarity for each memory entry (Eq. (6))
            dot_product = tf.matmul(tf.expand_dims(self._k_t, axis=0), M, transpose_b=True)
            dot_product = tf.squeeze(dot_product, axis=0)
            cosine_similarity = dot_product / (k_t_norm * M_norm + np.finfo(float).eps)

            # compute final weights (Eq. (5))
            wc_t = tf.nn.softmax(self._beta_t * cosine_similarity)

        return wc_t

    def _interpolation(self, wc_t):

        assert self._g_t is not None, 'Must call `emit_head_parameters` first.'

        with tf.variable_scope('interpolation'):
            # apply interpolation gate (Eq. (7))
            wg_t = self._g_t * wc_t + (1 - self._g_t) * self._w_last

        return wg_t

    def _convolutional_shift(self, wg_t):

        assert self._s_t is not None, 'Must call `emit_head_parameters` first.'

        with tf.variable_scope('convolutional_shift'):
            # apply circular convolution (Eq. (8))
            w_tilde_t = self._circular_convolution(wg_t, self._s_t)
            w_tilde_t = tf.reshape(w_tilde_t, shape=(self._memory_capacity,))

        return w_tilde_t

    def _sharpening(self, w_tilde_t):

        assert self._gamma_t is not None, 'Must call `emit_head_parameters` first.'

        with tf.variable_scope('sharpening'):
            # apply sharpening (Eq. (9))
            exponents = tf.ones_like(w_tilde_t) * self._gamma_t
            elevated = tf.pow(w_tilde_t, exponents)
            w_t = elevated / tf.reduce_sum(elevated)

        return w_t

    def _update_w_last(self, w_t):
        self._w_last = tf.assign(self._w_last, w_t)

    @staticmethod
    def _circular_convolution(v, k):
        """
        Computes circular convolution.
        Copy-pasted from [2].
        
        Args:
            v: a 1-D `Tensor` (vector)
            k: a 1-D `Tensor` (kernel)
        """
        size = int(v.get_shape()[0])
        kernel_size = int(k.get_shape()[0])
        kernel_shift = int(math.floor(kernel_size / 2.0))

        def loop(idx):
            if idx < 0:
                return size + idx
            if idx >= size:
                return idx - size
            else:
                return idx

        kernels = []
        for i in xrange(size):
            indices = [loop(i + j) for j in xrange(kernel_shift, -kernel_shift - 1, -1)]
            v_ = tf.gather(v, indices)
            kernels.append(tf.reduce_sum(v_ * k, 0))

        return tf.dynamic_stitch([i for i in xrange(size)], kernels)

    def reset(self):
        self._w_last = tf.assign(self._w_last,
                                 tf.ones(shape=(self._memory_capacity,), dtype=tf.float32) / self._memory_capacity)


class ReadHead(Head):

    def __init__(self, memory_capacity, memory_vector_size):
        super(ReadHead, self).__init__(memory_capacity, memory_vector_size)

    def read(self, M, w_t):

        self._update_w_last(w_t)

        w_t = tf.expand_dims(w_t, axis=0)
        r_t = tf.matmul(w_t, M)
        r_t = tf.squeeze(r_t, axis=0)

        return r_t


class WriteHead(Head):

    def __init__(self, memory_capacity, memory_vector_size):
        super(WriteHead, self).__init__(memory_capacity, memory_vector_size)

    def produce_memory_update(self, M, w_t, e_t, a_t):

        self._update_w_last(w_t)

        # update M
        M_tilde_t = M * (1 - tf.expand_dims(e_t, axis=-1))

        w_t = tf.expand_dims(w_t, axis=1)
        a_t = tf.tile(tf.expand_dims(a_t, axis=0), (self._memory_capacity, 1))
        memory_update = M_tilde_t + (w_t * a_t)

        return memory_update

    def produce_erase_vector(self, X):

        with tf.variable_scope('eraser'):
            X = tf.expand_dims(X, axis=0)
            e_t = tf.layers.dense(X, self._memory_capacity, activation=tf.nn.sigmoid)
            e_t = tf.squeeze(e_t, axis=0)

        return e_t

"""
References
----------
[1] Graves, Alex, Greg Wayne, and Ivo Danihelka. 
"Neural turing machines." arXiv preprint arXiv:1410.5401 (2014).
[2] https://github.com/carpedm20/NTM-tensorflow/blob/master/ops.py
"""
