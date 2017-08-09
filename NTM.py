import numpy as np
import tensorflow as tf
from os.path import exists
import os
import matplotlib.pyplot as plt

from heads import ReadHead, WriteHead
from utils import print_progress


class NTM(object):

    def __init__(self, memory_capacity, memory_vector_size, input_vector_size, sequence_length):

        self.memory_capacity = memory_capacity
        self.memory_vector_size = memory_vector_size
        self.input_vector_size = input_vector_size
        self.sequence_length = sequence_length

        self.memories = [tf.zeros(shape=(self.memory_capacity, self.memory_vector_size))]

        self.read_head = ReadHead(self.memory_capacity, self.memory_vector_size)
        self.write_head = WriteHead(self.memory_capacity, self.memory_vector_size)

        # graph (one placeholder/tensor for each sequence step)
        self.inputs = [tf.placeholder(shape=(self.input_vector_size,),
                                      dtype=tf.float32,
                                      name='input_vector_{:03d}'.format(i)) for i in range(0, self.sequence_length)]
        self.targets = [tf.placeholder(shape=(self.input_vector_size,),
                                       dtype=tf.float32,
                                       name='target_vector_{:03d}'.format(i)) for i in range(0, self.sequence_length)]
        self.outputs = []

        self.a_t = []  # write encodings
        self.ww_t = []  # write locations
        self.e_t = []  # write erase vectors

        self.rw_t = []  # read locations
        self.r_t = []  # read vectors

        self._build_graph()

        # optimization stuff (one for each sequence step)
        self.costs = []
        self.optimizers = []
        self.opt_steps = []
        self._build_optimization()

        # session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # summaries
        self.loss_summarization = None
        self.summarization = None
        self.train_writer = None

        # self.summaries = dict()
        # self.summaries['loss'] = tf.summary.scalar('loss', tf.reduce_mean(tf.stack(self.costs)))
        # self.summaries['inputs'] = tf.summary.image('inputs', tf.expand_dims(tf.expand_dims(tf.stack(self.inputs, axis=-1), axis=0), axis=-1))
        # self.summaries['targets'] = tf.summary.image('targets', tf.expand_dims(tf.expand_dims(tf.stack(self.targets, axis=-1), axis=0), axis=-1))
        # self.summaries['outputs'] = tf.summary.image('outputs', tf.expand_dims(tf.expand_dims(tf.stack(self.outputs, axis=-1), axis=0), axis=-1))
        # self.summaries['memory'] = tf.summary.image('memory', tf.expand_dims(tf.expand_dims(self.memory, axis=0), axis=-1))
        # self.summaries['a_t'] = tf.summary.image('a_t', tf.expand_dims(tf.expand_dims(tf.stack(self.a_t, axis=-1), axis=0), axis=-1))
        # self.summaries['ww_t'] = tf.summary.image('ww_t', tf.expand_dims(tf.expand_dims(tf.stack(self.ww_t, axis=-1), axis=0), axis=-1))
        # self.summaries['e_t'] = tf.summary.image('e_t', tf.expand_dims(tf.expand_dims(tf.stack(self.e_t, axis=-1), axis=0), axis=-1))
        # self.summaries['rw_t'] = tf.summary.image('rw_t', tf.expand_dims(tf.expand_dims(tf.stack(self.rw_t, axis=-1), axis=0), axis=-1))
        # self.summaries['r_t'] = tf.summary.image('r_t', tf.expand_dims(tf.expand_dims(tf.stack(self.r_t, axis=-1), axis=0), axis=-1))
        #
        # self.loss_summarization = tf.summary.merge([self.summaries['loss']])
        # self.summarization = tf.summary.merge_all()

        logs_dir = 'logs'
        if not exists(logs_dir):
            os.makedirs(logs_dir)
        # self.train_writer = tf.summary.FileWriter(logs_dir)

        self.global_step = 0

    def _build_graph(self):

        print('[*] Building a Neural Turing Machine.')
        for i in range(0, self.sequence_length):
            print_progress(float(i+1) / self.sequence_length)

            reuse = None if i == 0 else True

            cat = tf.concat([self.inputs[i]] + self.outputs, axis=0)
            controller_out = tf.layers.dense(tf.expand_dims(cat, axis=0), units=self.input_vector_size)
            controller_out = tf.squeeze(controller_out, axis=0)

            # write
            with tf.variable_scope('write', reuse=reuse):
                self.a_t.append(self._encode_input(controller_out))
                self.ww_t.append(self.write_head.produce_address(controller_out, self.memories[i]))
                self.e_t.append(self.write_head.produce_erase_vector(controller_out))
                memory_update = self.write_head.produce_memory_update(self.memories[i],
                                                                      self.ww_t[-1],
                                                                      self.e_t[-1],
                                                                      self.a_t[-1])
                self.memories.append(memory_update)

            # read
            with tf.variable_scope('read', reuse=reuse):
                self.rw_t.append(self.read_head.produce_address(controller_out, self.memories[i]))
                self.r_t.append(self.read_head.read(self.memories[i], self.rw_t[-1]))
                self.outputs.append(self._decode_output(self.r_t[-1]))

    def _build_optimization(self):

        print('[*] Building optimization problem.')
        self.loss = 0
        for i in range(0, self.sequence_length):
            print_progress(float(i+1) / self.sequence_length)

            self.costs.append(self.bce(self.targets[i], self.outputs[i]))
            self.loss += self.costs[-1]
            # self.optimizers.append(tf.train.RMSPropOptimizer(learning_rate=1e-4))
            # self.opt_steps.append(self.optimizers[-1].minimize(self.costs[-1]))

        self.opt = tf.train.AdamOptimizer(learning_rate=1e-2)
        self.opt_step = self.opt.minimize(self.loss)

    def _encode_input(self, I):

        with tf.variable_scope('input_encoder'):
            I = tf.expand_dims(I, axis=0)  # tf does not accept rank<2 for tf.layers.dense
            a_t = tf.layers.dense(I, self.memory_vector_size, activation=tf.nn.sigmoid)
            a_t = tf.squeeze(a_t, axis=0)  # squeeze fake dimension

        return a_t

    def _decode_output(self, r_t):

        with tf.variable_scope('output_decoder'):
            r_t = tf.expand_dims(r_t, axis=0)  # tf does not accept rank<2 for tf.layers.dense
            o_t = tf.layers.dense(r_t, self.input_vector_size, activation=tf.nn.sigmoid)
            o_t = tf.squeeze(o_t, axis=0)

        return o_t

    def reset(self):
        self.read_head.reset()
        self.write_head.reset()

    def bce(self, y_true, y_pred):

        return tf.reduce_sum(-(y_true * tf.log(y_pred + np.finfo(float).eps) + (1 - y_true) * tf.log(1 - y_pred + np.finfo(float).eps)))

    def train_on_sample(self, X, Y):

        assert X.shape[0] == Y.shape[0] == self.sequence_length, 'dimension mismatch.'

        feed_dict = {input_: vec for vec, input_ in zip(X, self.inputs)}
        feed_dict.update(
            {true_output: vec for vec, true_output in zip(Y, self.targets)}
        )

        fetches = {
            'a_t': [self.a_t[i] for i in range(0, self.sequence_length)],
            'ww_t': [self.ww_t[i] for i in range(0, self.sequence_length)],
            'e_t': [self.e_t[i] for i in range(0, self.sequence_length)],
            'rw_t': [self.rw_t[i] for i in range(0, self.sequence_length)],
            'r_t': [self.r_t[i] for i in range(0, self.sequence_length)],
            'output': [self.outputs[i] for i in range(0, self.sequence_length)],
            'memory': [self.memories[i] for i in range(0, self.sequence_length)],
            'loss': self.loss,
            'opt_step': self.opt_step
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

        return loss, a_t, ww_t, e_t, rw_t, r_t, out, memory


if __name__ == '__main__':

    model = NTM(512, 256, 100, 20)
    model.reset()

    all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print(all_variables)

    import numpy as np
    trainable_parameters = sum([np.prod([int(i) for i in v.get_shape()]) for v in all_variables])
    print('Total number of parameters: {:,}'.format(trainable_parameters))
