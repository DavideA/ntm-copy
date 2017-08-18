import numpy as np
import tensorflow as tf
import math


class Head(object):
    """
    This class implements a generic head, to 
    interact with the NTM memory by reading or writing
    upon it.
    """

    def __init__(self, memory_capacity, memory_vector_size, max_shift=3):
        """
        Builds a generic head.
        
        Parameters
        ----------
        memory_capacity:
            the number of vectors that can be stored in memory.
        memory_vector_size: int
            the dimensionality of the memory vector.
        max_shift: int
            the maximum convolutional shift (see sec 3.3.2 of [1]).
        """

        self._memory_capacity = memory_capacity
        self._memory_vector_size = memory_vector_size
        self._max_shift = max_shift

        # addressing emissions
        self._k_t = None
        self._beta_t = None
        self._g_t = None
        self._s_t = None
        self._gamma_t = None

        # addresses
        self._w_last = [tf.zeros(shape=(self._memory_capacity,), dtype=tf.float32)]

    def produce_address(self, controller_out, M):
        """
        This function implements the addressing mechanism of a NTM head.
        The addressing mechanism is described in its whole in sec 3.3 of [1].
        
        Parameters
        ----------
        controller_out: Tensor
            the output of the controller module.
        M: Tensor
            the current snapshot of the memory.

        Returns
        -------
        w_t: Tensor
            the weights for addressing the memory.
        """

        # emit controller parameters for addressing
        self._k_t, self._beta_t, self._g_t, self._s_t, self._gamma_t = self._emit_heads_parameters(controller_out)

        with tf.variable_scope('addressing_mechanism'):
            # flow of addressing mechanism
            wc_t = self._content_addressing(M)
            wg_t = self._interpolation(wc_t)
            w_tilde_t = self._convolutional_shift(wg_t)
            w_t = self._sharpening(w_tilde_t)

        self._w_last.append(w_t)

        return w_t

    def _content_addressing(self, M):
        """
        Provides content-based addressing.
        See sec 3.3.1 of [1].
        
        Parameters
        ----------
        M: Tensor
            the current snapshot of the memory.

        Returns
        -------
        w_t: Tensor
            the weights produced by content addressing.
        """

        assert self._k_t is not None and self._beta_t is not None, 'Must call `emit_head_parameters` first.'

        with tf.variable_scope('content_addressing'):
            # compute norm of key and memory content
            k_t_norm = 1#tf.sqrt(tf.reduce_sum(tf.pow(self._k_t, 2)))
            M_norm = 1#tf.sqrt(tf.reduce_sum(tf.pow(M, 2), axis=1))

            # compute cosine similarity for each memory entry (Eq. (6))
            dot_product = tf.matmul(tf.expand_dims(self._k_t, axis=0), M, transpose_b=True)
            dot_product = tf.squeeze(dot_product, axis=0)
            cosine_similarity = dot_product / (k_t_norm * M_norm + np.finfo(float).eps)

            # compute final weights (Eq. (5))
            wc_t = tf.nn.softmax(self._beta_t * cosine_similarity)

        return wc_t

    def _interpolation(self, wc_t):
        """
        Interpolates content-addressed weights with the gating vector.
        See sec 3.3.2 of [1].
        
        Parameters
        ----------
        wc_t: Tensor
            the weights produced by content addressing.
        Returns
        -------
        wg_t: Tensor
            the weights produced by the interpolation.
        """

        assert self._g_t is not None, 'Must call `emit_head_parameters` first.'

        with tf.variable_scope('interpolation'):
            # apply interpolation gate (Eq. (7))
            wg_t = self._g_t * wc_t + (1 - self._g_t) * self._w_last[-1]

        return wg_t

    def _convolutional_shift(self, wg_t):
        """
        Applies a circular convolution to a weight vector with the
        produced kernel s_t. See sec 3.3.2 of [1].
        
        Parameters
        ----------
        wg_t: Tensor
            the weights produced by the interpolation module.

        Returns
        -------
        w_tilde_t: Tensor
            the weights after the circular shift.
        """

        assert self._s_t is not None, 'Must call `emit_head_parameters` first.'

        with tf.variable_scope('convolutional_shift'):
            # apply circular convolution (Eq. (8))
            w_tilde_t = self._circular_convolution(wg_t, self._s_t)
            w_tilde_t = tf.reshape(w_tilde_t, shape=(self._memory_capacity,))

        return w_tilde_t

    def _sharpening(self, w_tilde_t):
        """
        Sharpens a weight vector of a sharpen factor gamma_t.
        See sec 3.3.2 of [1].
        
        Parameters
        ----------
        w_tilde_t: Tensor
            the weights produced by the convolutional shift.

        Returns
        -------
        w_t: Tensor
            the weights sharpened by gamma_t.
        """

        assert self._gamma_t is not None, 'Must call `emit_head_parameters` first.'

        with tf.variable_scope('sharpening'):
            # apply sharpening (Eq. (9))
            exponents = tf.ones_like(w_tilde_t) * self._gamma_t
            elevated = tf.pow(w_tilde_t, exponents)
            w_t = elevated / tf.reduce_sum(elevated)

        return w_t

    @staticmethod
    def _circular_convolution(v, k):
        """
        Computes circular convolution.
        Copy-pasted from [2], minimized and never opened again.
        
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

    def _emit_heads_parameters(self, controller_out):
        """
        Emits the parameters for addressing given the 
        output of the controller module.
        
        Parameters
        ----------
        controller_out: Tensor
            the output of the controller module.

        Returns
        -------
        tuple
            (k_t, beta_t, g_t, s_t, gamma_y)
        """
        with tf.variable_scope('head_parameters'):

            k_t = self._emit_k_t(controller_out)
            beta_t = self._emit_beta_t(controller_out)
            g_t = self._emit_g_t(controller_out)
            s_t = self._emit_s_t(controller_out)
            gamma_y = self._emit_gamma_t(controller_out)

        return k_t, beta_t, g_t, s_t, gamma_y

    def _emit_k_t(self, controller_out):
        """
        Provides k_t given the controller output.
        See sec 3.3.1 of [1].
        
        Parameters
        ----------
        controller_out: Tensor
            the controller output.

        Returns
        -------
        k_t: Tensor
            k_t used for content-based addressing
        """

        with tf.variable_scope('k_t_emitter'):
            controller_out = tf.expand_dims(controller_out, axis=0)
            k_t = tf.layers.dense(controller_out, self._memory_vector_size, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                  bias_initializer=tf.random_normal_initializer(stddev=0.5))
            k_t = tf.squeeze(k_t, axis=0)

        return k_t

    @staticmethod
    def _emit_beta_t(controller_out):
        """
        Provides beta_t given the controller output.
        See sec 3.3.1 of [1].
        
        Parameters
        ----------
        controller_out: Tensor
            the controller output.

        Returns
        -------
        beta_t: Tensor
            beta_t used for content-based addressing.
        """

        with tf.variable_scope('beta_t_emitter'):
            controller_out = tf.expand_dims(controller_out, axis=0)
            beta_t = tf.layers.dense(controller_out, 1, activation=tf.nn.softplus,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                     bias_initializer=tf.random_normal_initializer(stddev=0.5))
            beta_t = tf.squeeze(beta_t, axis=0)

        return beta_t

    @staticmethod
    def _emit_g_t(controller_out):
        """
        Provides g_t given the controller output.
        See sec 3.3.2 of [1].

        Parameters
        ----------
        controller_out: Tensor
            the controller output.

        Returns
        -------
        g_t: Tensor
            g_t used for location-based addressing.
        """

        with tf.variable_scope('g_t_emitter'):
            controller_out = tf.expand_dims(controller_out, axis=0)
            g_t = tf.layers.dense(controller_out, 1, activation=tf.nn.sigmoid,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                  bias_initializer=tf.random_normal_initializer(stddev=0.5))
            g_t = tf.squeeze(g_t, axis=0)

        return g_t

    def _emit_s_t(self, controller_out):
        """
        Provides s_t given the controller output.
        See sec 3.3.2 of [1].

        Parameters
        ----------
        controller_out: Tensor
            the controller output.

        Returns
        -------
        s_t: Tensor
            s_t used for location-based addressing.
        """

        with tf.variable_scope('s_t_emitter'):
            controller_out = tf.expand_dims(controller_out, axis=0)
            s_t = tf.layers.dense(controller_out, self._max_shift, activation=tf.nn.softmax,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                  bias_initializer=tf.random_normal_initializer(stddev=0.5))
            s_t = tf.squeeze(s_t, axis=0)

        return s_t

    @staticmethod
    def _emit_gamma_t(controller_out):
        """
        Provides gamma_t given the controller output.
        See sec 3.3.2 of [1].

        Parameters
        ----------
        controller_out: Tensor
            the controller output.

        Returns
        -------
        gamma_t: Tensor
            gamma_t used for location-based addressing.
        """

        with tf.variable_scope('gamma_t_emitter'):
            controller_out = tf.expand_dims(controller_out, axis=0)
            gamma_t = tf.layers.dense(controller_out, 1, activation=tf.nn.softplus,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                      bias_initializer=tf.random_normal_initializer(stddev=0.5))
            gamma_t = tf.add(gamma_t, tf.constant(1.0))
            gamma_t = tf.squeeze(gamma_t, axis=0)

        return gamma_t


class ReadHead(Head):
    """
    Specifies a head for reading from the memory.
    """

    def __init__(self, memory_capacity, memory_vector_size, max_shift=3):
        """
        Builds a read head.

        Parameters
        ----------
        memory_capacity:
            the number of vectors that can be stored in memory.
        memory_vector_size: int
            the dimensionality of the memory vector.
        max_shift: int
            the maximum convolutional shift (see sec 3.3.2 of [1]).
        """

        super(ReadHead, self).__init__(memory_capacity, memory_vector_size, max_shift)

    @staticmethod
    def read(M, w_t):
        """
        Reads from the memory.
        See sec 3.1 of [1].
        
        Parameters
        ----------
        M: Tensor
            the current snapshot of the memory.
        w_t: Tensor
            the weight vector used for reading.

        Returns
        -------
        r_t: Tensor
            the vector read from memory.
        """

        # read (Eq (2))
        w_t = tf.expand_dims(w_t, axis=0)
        r_t = tf.matmul(w_t, M)
        r_t = tf.squeeze(r_t, axis=0)

        return r_t


class WriteHead(Head):
    """
    Specifies a head for writing onto the memory.
    """

    def __init__(self, memory_capacity, memory_vector_size, max_shift=3):
        """
        Builds a read head.

        Parameters
        ----------
        memory_capacity:
            the number of vectors that can be stored in memory.
        memory_vector_size: int
            the dimensionality of the memory vector.
        max_shift: int
            the maximum convolutional shift (see sec 3.3.2 of [1]).
        """

        super(WriteHead, self).__init__(memory_capacity, memory_vector_size, max_shift)

    def produce_memory_update(self, M, w_t, e_t, a_t):
        """
        Produces the next snapshot of the memory.
        See sec 3.2 of [1].
        
        Parameters
        ----------
        M: Tensor
            the current snapshot of the memory.
        w_t: Tensor
            the weight vector to write into.
        e_t: Tensor
            the erase vector.
        a_t: Tensor
            the add vector.

        Returns
        -------
        memory_update: Tensor
            the next memory snapshot.
        """

        # erase (eq. (3))
        M_tilde_t = M * (1 - tf.expand_dims(e_t, axis=-1))

        # add (eq. (4))
        w_t = tf.expand_dims(w_t, axis=1)
        a_t = tf.tile(tf.expand_dims(a_t, axis=0), (self._memory_capacity, 1))
        memory_update = M_tilde_t + (w_t * a_t)

        return memory_update

    def produce_add_vector(self, controller_out):
        """
        Produces an add vector given the controller output.
        See sec 3.2 of [1].
        
        Parameters
        ----------
        controller_out: Tensor
            the output of the controller.

        Returns
        -------
        a_t: Tensor
            the add vector.
        """

        with tf.variable_scope('add_vector_encoder'):
            controller_out = tf.expand_dims(controller_out, axis=0)  # tf does not accept rank<2 for tf.layers.dense
            a_t = tf.layers.dense(controller_out, self._memory_vector_size, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                  bias_initializer=tf.random_normal_initializer(stddev=0.5))
            a_t = tf.squeeze(a_t, axis=0)  # squeeze fake dimension

        return a_t

    def produce_erase_vector(self, controller_out):
        """
        Produces the erase vector given the controller output.
        See sec 3.2 of [1].
        
        Parameters
        ----------
        controller_out: Tensor
            the controller output.

        Returns
        -------
        e_t: Tensor
            the erase vector.
        """

        with tf.variable_scope('eraser'):
            controller_out = tf.expand_dims(controller_out, axis=0)
            e_t = tf.layers.dense(controller_out, self._memory_capacity, activation=tf.nn.sigmoid,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                  bias_initializer=tf.random_normal_initializer(stddev=0.5))
            e_t = tf.squeeze(e_t, axis=0)

        return e_t

"""
References
----------
[1] Graves, Alex, Greg Wayne, and Ivo Danihelka. 
"Neural turing machines." arXiv preprint arXiv:1410.5401 (2014).
[2] https://github.com/carpedm20/NTM-tensorflow/blob/master/ops.py
"""
