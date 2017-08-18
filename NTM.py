import numpy as np
import tensorflow as tf
from os.path import join, exists
import os

from heads import ReadHead, WriteHead
from utils import print_progress


class NTM(object):

    def __init__(self, memory_capacity, memory_vector_size, input_vector_size, sequence_length,
                 controller_dim=100):

        self.memory_capacity = memory_capacity
        self.memory_vector_size = memory_vector_size
        self.input_vector_size = input_vector_size
        self.sequence_length = sequence_length
        self.controller_dim = controller_dim

        self.memories = [tf.tanh(tf.random_normal(shape=(self.memory_capacity, self.memory_vector_size), stddev=0.5))]

        self.read_head = ReadHead(self.memory_capacity, self.memory_vector_size)
        self.write_head = WriteHead(self.memory_capacity, self.memory_vector_size)

        # graph (one placeholder/tensor for each sequence step)
        self.inputs = [tf.placeholder(shape=(self.input_vector_size,),
                                      dtype=tf.float32,
                                      name='input_vector_{:03d}'.format(i)) for i in range(0, self.sequence_length)]
        self.targets = [tf.placeholder(shape=(self.input_vector_size,),
                                       dtype=tf.float32,
                                       name='target_vector_{:03d}'.format(i)) for i in range(0, self.sequence_length)]
        self.start_token = tf.placeholder(shape=(self.input_vector_size,), dtype=tf.float32, name='start_token')
        self.end_token = tf.placeholder(shape=(self.input_vector_size,), dtype=tf.float32, name='end_token')
        self.zeros = tf.zeros(shape=(self.input_vector_size,), dtype=tf.float32, name='zeros')
        self.outputs = []

        self.a_t = [tf.zeros(shape=(self.memory_vector_size,), dtype=tf.float32)]  # write encodings
        self.ww_t = [tf.nn.softmax(tf.range(self.memory_capacity, 0, -1, dtype=tf.float32))]  # write locations
        self.e_t = [tf.zeros(shape=(self.memory_capacity,), dtype=tf.float32)]  # write erase vectors

        self.rw_t = [tf.nn.softmax(tf.range(self.memory_capacity, 0, -1, dtype=tf.float32))]  # read locations
        self.r_t = [tf.tanh(tf.random_normal(shape=(self.memory_vector_size,), stddev=0.5))]  # read vectors

        self._build_graph()

        # optimization stuff (one for each sequence step)
        self.opt_step = None
        self.loss = 0
        self._build_optimization()

        # session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # summaries
        self.loss_summarization = None
        self.summarization = None
        self.train_writer = None
        self.summaries = dict()

        self._make_summaries()

        self.global_step = 0

    def _build_graph(self):

        print('[*] Building a Neural Turing Machine.')

        # present start token
        controller_out = self._controller(self.start_token, self.r_t[0], reuse=None)
        self._read_write(controller_out, reuse=None)

        # present inputs
        print('Input chain: ')
        for i in range(0, self.sequence_length):
            print_progress(float(i+1) / self.sequence_length)

            controller_out = self._controller(self.inputs[i], self.r_t[-1], reuse=True)
            self._read_write(controller_out, reuse=True)

        # present end token
        controller_out = self._controller(self.end_token, self.r_t[-1], reuse=True)
        self._read_write(controller_out, reuse=True)

        # present outputs
        print('Output chain: ')
        for i in range(0, self.sequence_length):
            print_progress(float(i + 1) / self.sequence_length)

            controller_out = self._controller(self.zeros, self.r_t[-1], reuse=True)
            self._read_write(controller_out, reuse=True)

            reuse = None if i == 0 else True
            self.outputs.append(self._decode_output(self.r_t[-1], reuse=reuse))
        print('Done.')

    def _controller(self, cur_input, last_read, reuse):

        with tf.variable_scope('controller', reuse=reuse):
            cat = tf.concat([cur_input, last_read], axis=0)
            controller_h = tf.layers.dense(tf.expand_dims(cat, axis=0), units=256, activation=tf.nn.sigmoid,
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                           bias_initializer=tf.random_normal_initializer(stddev=0.5))
            controller_out = tf.layers.dense(controller_h, units=self.controller_dim, activation=tf.nn.sigmoid,
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                             bias_initializer=tf.random_normal_initializer(stddev=0.5))

        return tf.squeeze(controller_out, axis=0)

    def _read_write(self, controller_out, reuse):

        # read
        with tf.variable_scope('read', reuse=reuse):
            self.rw_t.append(self.read_head.produce_address(controller_out, self.memories[-1]))
            self.r_t.append(self.read_head.read(self.memories[-1], self.rw_t[-1]))

        # write
        with tf.variable_scope('write', reuse=reuse):
            self.a_t.append(self._encode_input(controller_out))
            self.ww_t.append(self.write_head.produce_address(controller_out, self.memories[-1]))
            self.e_t.append(self.write_head.produce_erase_vector(controller_out))
            memory_update = self.write_head.produce_memory_update(self.memories[-1],
                                                                  self.ww_t[-1],
                                                                  self.e_t[-1],
                                                                  self.a_t[-1])
            self.memories.append(memory_update)

    def _build_optimization(self):

        print('[*] Building optimization problem.')
        for i in range(0, self.sequence_length):
            print_progress(float(i+1) / self.sequence_length)

            self.loss += self.bce(self.targets[i], self.outputs[i])

        self.opt = tf.train.RMSPropOptimizer(learning_rate=1e-4)
        # self.opt_step = self.opt.minimize(self.loss)
        gvs = self.opt.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
        self.opt_step = self.opt.apply_gradients(capped_gvs)

    def _encode_input(self, I):

        with tf.variable_scope('input_encoder'):
            I = tf.expand_dims(I, axis=0)  # tf does not accept rank<2 for tf.layers.dense
            a_t = tf.layers.dense(I, self.memory_vector_size, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                  bias_initializer=tf.random_normal_initializer(stddev=0.5))
            a_t = tf.squeeze(a_t, axis=0)  # squeeze fake dimension

        return a_t

    def _decode_output(self, r_t, reuse):

        with tf.variable_scope('output_decoder', reuse=reuse):
            r_t = tf.expand_dims(r_t, axis=0)  # tf does not accept rank<2 for tf.layers.dense
            o_t = tf.layers.dense(r_t, self.input_vector_size, activation=tf.nn.sigmoid,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                  bias_initializer=tf.random_normal_initializer(stddev=0.5))
            o_t = tf.squeeze(o_t, axis=0)

        return o_t

    def bce(self, y_true, y_pred):

        return tf.reduce_mean(-(y_true * tf.log(y_pred + np.finfo(float).eps) + (1 - y_true) * tf.log(1 - y_pred + np.finfo(float).eps)))

    def _make_summaries(self):

        self.summaries['loss'] = tf.summary.scalar('loss', self.loss)

        self.summaries['0_inputs_outputs'] = \
            tf.summary.image('0_inputs_outputs', tf.concat([
                tf.expand_dims(tf.expand_dims(tf.stack(self.inputs, axis=-1), axis=0), axis=-1),
                tf.ones(shape=(1, self.input_vector_size, 1, 1)),  # white separator
                tf.expand_dims(tf.expand_dims(tf.stack(self.outputs, axis=-1), axis=0), axis=-1)],
                axis=2))

        self.summaries['1_write_and_read_locations'] = \
            tf.summary.image('1_write_and_read_locations', tf.concat([
                tf.expand_dims(tf.expand_dims(tf.stack(self.ww_t, axis=-1), axis=0), axis=-1),
                tf.ones(shape=(1, self.memory_capacity, 1, 1)),  # white separator
                tf.expand_dims(tf.expand_dims(tf.stack(self.rw_t, axis=-1), axis=0), axis=-1)],
                axis=2))

        self.summaries['2_memory'] = \
            tf.summary.image('2_memory', tf.expand_dims(tf.expand_dims(self.memories[-1], axis=0), axis=-1))

        self.summaries['3_add_read_vectors'] = \
            tf.summary.image('3_add_read_vectors', tf.concat([
                tf.expand_dims(tf.expand_dims(tf.stack(self.a_t, axis=-1), axis=0), axis=-1),
                tf.ones(shape=(1, self.memory_vector_size, 1, 1)),  # white separator
                tf.expand_dims(tf.expand_dims(tf.stack(self.r_t, axis=-1), axis=0), axis=-1)],
                axis=2))

        self.summaries['4_erase_vectors'] = \
            tf.summary.image('4_erase_vectors',
                             tf.expand_dims(tf.expand_dims(tf.stack(self.e_t, axis=-1), axis=0), axis=-1))

        self.loss_summarization = tf.summary.merge([self.summaries['loss']])
        self.summarization = tf.summary.merge_all()

        logs_dir = join('logs', 'seq_{:02d}'.format(self.sequence_length))
        if not exists(logs_dir):
            os.makedirs(logs_dir)
        self.train_writer = tf.summary.FileWriter(logs_dir, self.sess.graph)

    def train_on_sample(self, start, X, end, Y):

        assert X.shape[0] == Y.shape[0] == self.sequence_length, 'dimension mismatch.'

        feed_dict = {input_: vec for vec, input_ in zip(X, self.inputs)}
        feed_dict.update(
            {true_output: vec for vec, true_output in zip(Y, self.targets)}
        )
        feed_dict.update(
            {
                self.start_token: start,
                self.end_token: end
            }
        )

        fetches = {
            'a_t': self.a_t,
            'ww_t': self.ww_t,
            'e_t': self.e_t,
            'rw_t': self.rw_t,
            'r_t': self.r_t,
            'output': self.outputs,
            'memory': self.memories,
            'loss': self.loss,
            'opt_step': self.opt_step,
        }

        res = self.sess.run(fetches=fetches, feed_dict=feed_dict)

        self.global_step += 1

        loss = res['loss']
        a_t = np.stack(res['a_t'], axis=0)
        ww_t = np.stack(res['ww_t'], axis=0)
        e_t = np.stack(res['e_t'], axis=0)
        rw_t = np.stack(res['rw_t'], axis=0)
        r_t = np.stack(res['r_t'], axis=0)
        out = np.stack(res['output'], axis=0)
        memory = res['memory']

        # write summary for loss
        self.train_writer.add_summary(self.sess.run(self.loss_summarization, feed_dict=feed_dict),
                                      global_step=self.global_step)

        # write summary for images
        if self.global_step % 200 == 0:
            self.train_writer.add_summary(self.sess.run(self.summarization, feed_dict=feed_dict),
                                          global_step=self.global_step)

        return loss, a_t, ww_t, e_t, rw_t, r_t, out, memory


if __name__ == '__main__':

    model = NTM(512, 256, 10, 3)

    all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print(all_variables)

    import numpy as np
    trainable_parameters = sum([np.prod([int(i) for i in v.get_shape()]) for v in all_variables])
    print('Total number of parameters: {:,}'.format(trainable_parameters))
