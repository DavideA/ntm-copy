import tensorflow as tf


class Controller(object):
    """
    A NTM controller. Its purpose is to produce a feature vector
    given the current input token and the last read vector, 
    that will be further used by heads to address and interact
    with the memory.
    """

    def __init__(self, output_dim, hidden_dim=256):
        """
        Builds a simple NTM controller.
        
        Parameters
        ----------
        output_dim: int
            the dimensionality of the output feature vector.
        hidden_dim: int
            the dimensionality of the hidden layer.
        """

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def emit_feature_vector(self, cur_input, last_read, reuse):
        """
        Provides a controller vector given the current input and
        the last vector read from the memory.
        
        Parameters
        ----------
        cur_input: Tensor
            the current input token.
        last_read: Tensor
            the last vector read from the memory.
        reuse:
            whether or not to reuse variables. Either None or True.

        Returns
        -------
        controller_out: Tensor
            the controller feature vector.
        """

        with tf.variable_scope('controller', reuse=reuse):
            cat = tf.concat([cur_input, last_read], axis=0)
            controller_h = tf.layers.dense(tf.expand_dims(cat, axis=0), units=self.hidden_dim, activation=tf.nn.sigmoid,
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                           bias_initializer=tf.random_normal_initializer(stddev=0.5))
            controller_out = tf.layers.dense(controller_h, units=self.output_dim, activation=tf.nn.sigmoid,
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                             bias_initializer=tf.random_normal_initializer(stddev=0.5))
            controller_out = tf.squeeze(controller_out, axis=0)

        return controller_out
