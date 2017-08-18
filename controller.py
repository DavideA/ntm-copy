import tensorflow as tf


class Controller(object):

    def __init__(self, memory_capacity, memory_vector_size, max_shift=None):

        self.memory_capacity = memory_capacity
        self.memory_vector_size = memory_vector_size

        if max_shift is None:
            self.max_shift = self.memory_capacity - 1
        else:
            self.max_shift = max_shift

    def emit_heads_parameters(self, X):

        with tf.variable_scope('controller'):

            k_t = self.emit_k_t(X)
            beta_t = self.emit_beta_t(X)
            g_t = self.emit_g_t(X)
            s_t = self.emit_s_t(X)
            gamma_y = self.emit_gamma_t(X)

        return k_t, beta_t, g_t, s_t, gamma_y

    def emit_k_t(self, X):

        with tf.variable_scope('k_t_emitter'):
            X = tf.expand_dims(X, axis=0)
            k_t = tf.layers.dense(X, self.memory_vector_size, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                  bias_initializer=tf.random_normal_initializer(stddev=0.5))
            k_t = tf.squeeze(k_t, axis=0)

        return k_t

    def emit_beta_t(self, X):

        with tf.variable_scope('beta_t_emitter'):
            X = tf.expand_dims(X, axis=0)
            beta_t = tf.layers.dense(X, 1, activation=tf.nn.softplus,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                     bias_initializer=tf.random_normal_initializer(stddev=0.5))
            beta_t = tf.squeeze(beta_t, axis=0)

        return beta_t

    def emit_g_t(self, X):

        with tf.variable_scope('g_t_emitter'):
            X = tf.expand_dims(X, axis=0)
            g_t = tf.layers.dense(X, 1, activation=tf.nn.sigmoid,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                  bias_initializer=tf.random_normal_initializer(stddev=0.5))
            g_t = tf.squeeze(g_t, axis=0)

        return g_t

    def emit_s_t(self, X):

        with tf.variable_scope('s_t_emitter'):
            X = tf.expand_dims(X, axis=0)
            s_t = tf.layers.dense(X, self.max_shift, activation=tf.nn.softmax,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                  bias_initializer=tf.random_normal_initializer(stddev=0.5))
            s_t = tf.squeeze(s_t, axis=0)

        return s_t

    def emit_gamma_t(self, X):

        with tf.variable_scope('gamma_t_emitter'):
            X = tf.expand_dims(X, axis=0)
            gamma_t = tf.layers.dense(X, 1, activation=tf.nn.softplus,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                      bias_initializer=tf.random_normal_initializer(stddev=0.5))
            gamma_t = tf.add(gamma_t, tf.constant(1.0))
            gamma_t = tf.squeeze(gamma_t, axis=0)

        return gamma_t
