import numpy as np
import tensorflow as tf
from os.path import exists
import os

from heads import ReadHead, WriteHead


class NTM(object):

    def __init__(self, memory_capacity, memory_vector_size, input_vector_size):

        self.memory_capacity = memory_capacity
        self.memory_vector_size = memory_vector_size
        self.input_vector_size = input_vector_size

        self.memory = tf.Variable(initial_value=tf.zeros(shape=(self.memory_capacity, self.memory_vector_size),
                                                         dtype=tf.float32,
                                                         name='NTM_memory'),
                                  trainable=False)

        self.read_head = ReadHead(self.memory_capacity, self.memory_vector_size)
        self.write_head = WriteHead(self.memory_capacity, self.memory_vector_size)

        self.input = tf.placeholder(shape=(None, self.input_vector_size), dtype=tf.float32, name='input_vector')
        self.target = tf.placeholder(shape=(None, self.input_vector_size), dtype=tf.float32, name='target_vector')
        self.output = tf.map_fn(lambda x: self._forward(x), self.input, name='output_vector')

        # optimization stuff
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4, momentum=0.9, decay=0.95)
        self.cost = self.bce(self.target, self.output)
        self.train_fetches = {'cost': self.cost,
                              'opt_step': self.optimizer.minimize(self.cost),
                              'out': self.output}

        # session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # summaries
        self.loss_summarization = None
        self.summarization = None
        self.train_writer = None

        self.summaries = dict()
        self.summaries['loss'] = tf.summary.scalar('loss', self.cost)
        self.summaries['input'] = tf.summary.image('input', tf.expand_dims(tf.expand_dims(self.input, axis=0), axis=-1))
        self.summaries['output'] = tf.summary.image('output', tf.expand_dims(tf.expand_dims(self.output, axis=0), axis=-1))
        self.summaries['target'] = tf.summary.image('target', tf.expand_dims(tf.expand_dims(self.target, axis=0), axis=-1))
        self.summaries['memory'] = tf.summary.image('memory', tf.expand_dims(tf.expand_dims(self.memory, axis=0), axis=-1))

        self.loss_summarization = tf.summary.merge([self.summaries['loss']])
        self.summarization = tf.summary.merge_all()

        logs_dir = 'logs'
        if not exists(logs_dir):
            os.makedirs(logs_dir)
        self.train_writer = tf.summary.FileWriter(logs_dir)

        self.global_step = 0

    def _forward(self, X):

        # write
        with tf.variable_scope('write'):
            a_t = self._encode_input(X)
            w_t = self.write_head.produce_address(X, self.memory)
            e_t = self.write_head.produce_erase_vector(X)
            self.write_head.write(self.memory, w_t, e_t, a_t)

        # read
        with tf.variable_scope('read'):
            w_t = self.read_head.produce_address(X, self.memory)
            r_t = self.read_head.read(self.memory, w_t)
            o_t = self._decode_output(r_t)

        return o_t

    def _encode_input(self, I):

        with tf.variable_scope('input_encoder'):
            I = tf.expand_dims(I, axis=0)  # tf does not accept rank<2 for tf.layers.dense
            a_t = tf.layers.dense(I, self.memory_vector_size, activation=tf.nn.tanh)
            a_t = tf.squeeze(a_t, axis=0)  # squeeze fake dimension

        return a_t

    def _decode_output(self, r_t):

        with tf.variable_scope('output_decoder'):
            r_t = tf.expand_dims(r_t, axis=0)  # tf does not accept rank<2 for tf.layers.dense
            o_t = tf.layers.dense(r_t, self.input_vector_size, activation=tf.nn.sigmoid)
            o_t = tf.squeeze(o_t, axis=0)

        return o_t

    def reset(self):
        tf.assign(self.memory, tf.zeros(shape=(self.memory_capacity, self.memory_vector_size),
                                        dtype=tf.float32,
                                        name='NTM_memory'))
        self.read_head.reset()
        self.write_head.reset()

    def bce(self, y_true, y_pred):

        return tf.reduce_sum(-(y_true * tf.log(y_pred + np.finfo(float).eps) + (1 - y_true) * tf.log(1 - y_pred + np.finfo(float).eps)))

    def train_on_sample(self, X, Y):

        feed_dict = {
            self.input: X,
            self.target: Y
        }

        res = self.sess.run(self.train_fetches, feed_dict=feed_dict)
        print 'Cost: {},\tout_boundaries:{},{}'.format(res['cost'], res['out'].max(), res['out'].min())

        # add summaries
        self.train_writer.add_summary(self.sess.run(self.loss_summarization, feed_dict),
                                      global_step=self.global_step)

        # write summary for images
        if self.global_step % 200 == 0:
            self.train_writer.add_summary(self.sess.run(self.summarization, feed_dict),
                                          global_step=self.global_step)

        self.global_step += 1


if __name__ == '__main__':

    model = NTM(512, 256, 100)
    model.reset()

    all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print(all_variables)

    import numpy as np
    trainable_parameters = sum([np.prod([int(i) for i in v.get_shape()]) for v in all_variables])
    print('Total number of parameters: {:,}'.format(trainable_parameters))
