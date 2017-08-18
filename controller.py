import tensorflow as tf


class Controller(object):

    def __init__(self, output_dim, hidden_dim=256):

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def emit_feature_vector(self, cur_input, last_read, reuse):
        with tf.variable_scope('controller', reuse=reuse):
            cat = tf.concat([cur_input, last_read], axis=0)
            controller_h = tf.layers.dense(tf.expand_dims(cat, axis=0), units=self.hidden_dim, activation=tf.nn.sigmoid,
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                           bias_initializer=tf.random_normal_initializer(stddev=0.5))
            controller_out = tf.layers.dense(controller_h, units=self.output_dim, activation=tf.nn.sigmoid,
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                             bias_initializer=tf.random_normal_initializer(stddev=0.5))

        return tf.squeeze(controller_out, axis=0)
