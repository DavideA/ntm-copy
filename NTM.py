import numpy as np
import tensorflow as tf
from os.path import join, exists
import os

from controller import Controller
from heads import ReadHead, WriteHead
from utils import print_progress


class NTM(object):
    """
    Models a naive implementation of a Neural Turing Machine.
    The implementation is the simplest you could write reading the original paper [1].
    Features:
        * a feed-forward 2-layer controller.
        * a single read head and a single write head.
    """

    def __init__(self, memory_capacity, memory_vector_size, input_vector_size, sequence_length,
                 controller_out_dim, controller_hidden_dim, learning_rate,
                 min_grad, max_grad, logdir):
        """ 
        Builds a Neural Turing Machine model.
        
        Parameters
        ----------
        memory_capacity: int
            the number of memory cells.
        memory_vector_size: int
            the dimensionality of each memory cell.
        input_vector_size: int
            the dimensionality of input vectors.
        sequence_length: int
            the length of the sequence to copy.
        controller_out_dim: int
            the dimensionality of the feature vector provided by the controller.
        controller_hidden_dim: int
            the dimensionality of the hidden layer of the controller.
        learning_rate: float
            the optimizer learning rate.
        min_grad: float
            the minimum value of the gradient for clipping.
        max_grad: float
            the maximum value of the gradient for clipping.
        logdir: str
            the directory where to store logs.
        """

        self.memory_capacity = memory_capacity
        self.memory_vector_size = memory_vector_size
        self.input_vector_size = input_vector_size
        self.sequence_length = sequence_length
        self.controller_out_dim = controller_out_dim
        self.controller_hidden_dim = controller_hidden_dim
        self.learning_rate = learning_rate
        self.min_grad = min_grad
        self.max_grad = max_grad
        self.logdir = logdir

        # build controller and heads
        self.controller = Controller(output_dim=self.controller_out_dim, hidden_dim=self.controller_hidden_dim)
        self.read_head = ReadHead(self.memory_capacity, self.memory_vector_size)
        self.write_head = WriteHead(self.memory_capacity, self.memory_vector_size)

        # graph (one placeholder/tensor for each sequence step)
        self.inputs = [tf.placeholder(shape=(self.input_vector_size,),
                                      dtype=tf.float32,
                                      name='input_vector_{:03d}'.format(t)) for t in range(0, self.sequence_length)]
        self.targets = [tf.placeholder(shape=(self.input_vector_size,),
                                       dtype=tf.float32,
                                       name='target_vector_{:03d}'.format(t)) for t in range(0, self.sequence_length)]
        self.start_token = tf.placeholder(shape=(self.input_vector_size,), dtype=tf.float32, name='start_token')
        self.end_token = tf.placeholder(shape=(self.input_vector_size,), dtype=tf.float32, name='end_token')
        self.zeros = tf.zeros(shape=(self.input_vector_size,), dtype=tf.float32, name='zeros')
        self.outputs = []

        self.memories = []  # the memory
        self.ww_t = []  # write locations
        self.a_t = []  # write add terms
        self.e_t = []  # write erase terms

        self.rw_t = []  # read locations
        self.r_t = []  # read vectors

        self._build_forward_graph()

        # optimization stuff
        self.opt_step = None
        self.loss = 0
        self._build_backward_graph()

        # session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.global_step = 0

        # summaries stuff
        self.loss_summarization = None
        self.summarization = None
        self.train_writer = None
        self.summaries = dict()

        self._make_summaries()

        # print how many parameters are there
        all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        trainable_parameters = sum([np.prod([int(i) for i in v.get_shape()]) for v in all_variables])
        print('Model built. Total number of parameters: {:,}'.format(trainable_parameters))

    def _build_forward_graph(self):
        """
        Builds the forward computation graph.
        It is made of many read/write cycles.
        """

        print('[*] Building a Neural Turing Machine.')

        self._initalize_state()

        # present start token
        controller_out = self.controller.emit_feature_vector(self.start_token, self.r_t[0], reuse=None)
        self._read_write(controller_out, reuse=None)

        # present inputs
        print('Input chain: ')
        for t in range(0, self.sequence_length):
            print_progress(float(t + 1) / self.sequence_length)

            controller_out = self.controller.emit_feature_vector(self.inputs[t], self.r_t[-1], reuse=True)
            self._read_write(controller_out, reuse=True)

        # present end token
        controller_out = self.controller.emit_feature_vector(self.end_token, self.r_t[-1], reuse=True)
        self._read_write(controller_out, reuse=True)

        # present outputs
        print('Output chain: ')
        for t in range(0, self.sequence_length):
            print_progress(float(t + 1) / self.sequence_length)

            controller_out = self.controller.emit_feature_vector(self.zeros, self.r_t[-1], reuse=True)
            self._read_write(controller_out, reuse=True)

            reuse = None if t == 0 else True
            self.outputs.append(self._decode_read_vector(self.r_t[-1], reuse=reuse))
        print('Done.')

    def _build_backward_graph(self):
        """
        Builds the backward graph for optimization.
        """

        print('[*] Building optimization problem.')
        with tf.variable_scope('optimization'):
            for t in range(0, self.sequence_length):
                print_progress(float(t+1) / self.sequence_length)

                # loss is a binary crossentropy for each timestep
                self.loss += self.bce(self.targets[t], self.outputs[t])

            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            g = self.opt.compute_gradients(self.loss)
            clipped_g = [(tf.clip_by_value(grad, self.min_grad, self.max_grad), var) for grad, var in g]
            self.opt_step = self.opt.apply_gradients(clipped_g)

    def _read_write(self, controller_out, reuse):
        """
        Performs a single-step read/write cycle.
        
        Parameters
        ----------
        controller_out: Tensor
            the output of the controller.
        reuse:
            whether or not to reuse variables. Either None or True.
        """

        # read
        with tf.variable_scope('read', reuse=reuse):
            self.rw_t.append(self.read_head.produce_address(controller_out, self.memories[-1]))
            self.r_t.append(self.read_head.read(self.memories[-1], self.rw_t[-1]))

        # write
        with tf.variable_scope('write', reuse=reuse):
            self.a_t.append(self.write_head.produce_add_vector(controller_out))
            self.ww_t.append(self.write_head.produce_address(controller_out, self.memories[-1]))
            self.e_t.append(self.write_head.produce_erase_vector(controller_out))
            memory_update = self.write_head.produce_memory_update(self.memories[-1],
                                                                  self.ww_t[-1],
                                                                  self.e_t[-1],
                                                                  self.a_t[-1])
            self.memories.append(memory_update)

    def _decode_read_vector(self, r_t, reuse):
        """
        Decodes the vector read from memory into 
        an output token.
        
        Parameters
        ----------
        r_t: Tensor
            the read vector.
        reuse:
            whether or not to reuse variables. Either None or True.

        Returns
        -------
        o_t: Tensor
            the current output token.
        """

        with tf.variable_scope('output_decoder', reuse=reuse):
            r_t = tf.expand_dims(r_t, axis=0)
            o_t = tf.layers.dense(r_t, self.input_vector_size, activation=tf.nn.sigmoid,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.5),
                                  bias_initializer=tf.random_normal_initializer(stddev=0.5))
            o_t = tf.squeeze(o_t, axis=0)

        return o_t

    def _initalize_state(self):
        """
        Initializes the state of the NTM.
        Turns out initialization is critic.
        Lots of the manual tuning involved the next 5 lines of code.
        """

        self.memories.append(tf.tanh(
            tf.random_normal(shape=(self.memory_capacity, self.memory_vector_size), stddev=0.5)))
        self.ww_t.append(tf.nn.softmax(tf.range(self.memory_capacity, 0, -1, dtype=tf.float32)))
        self.rw_t.append(tf.nn.softmax(tf.range(self.memory_capacity, 0, -1, dtype=tf.float32)))
        self.r_t.append(tf.tanh(tf.random_normal(shape=(self.memory_vector_size,), stddev=0.5)))

    @staticmethod
    def bce(y_true, y_pred):
        """
        Binary crossentropy.
        
        Parameters
        ----------
        y_true: Tensor
            groundtruth values.
        y_pred: Tensor
            predicted values.

        Returns
        -------
        Tensor
            the binary crossentropy.
        """
        return tf.reduce_mean(-(y_true * tf.log(y_pred + np.finfo(float).eps) +
                                (1 - y_true) * tf.log(1 - y_pred + np.finfo(float).eps)))

    def _make_summaries(self):
        """
        Function to build summary stuff.
        """

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

        logs_dir = join(self.logdir, 'seq_len_{:02d}'.format(self.sequence_length))

        # make summary dir
        if not exists(logs_dir):
            os.makedirs(logs_dir)
        self.train_writer = tf.summary.FileWriter(logs_dir, self.sess.graph)

    def train_on_sample(self, start_token, X, end_token, Y):
        """
        Function to train the NTM model on a single example.
        
        Parameters
        ----------
        start_token: ndarray
            the start token.
        X: ndarray
            the sequence to be copied.
        end_token: ndarray
            the end token.
        Y: ndarray
            the groundtruth value.
        """

        assert X.shape[0] == Y.shape[0] == self.sequence_length, 'dimension mismatch.'

        feed_dict = {input_: vec for vec, input_ in zip(X, self.inputs)}
        feed_dict.update(
            {true_output: vec for vec, true_output in zip(Y, self.targets)}
        )
        feed_dict.update(
            {
                self.start_token: start_token,
                self.end_token: end_token
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
            'loss_summarization': self.loss_summarization
        }

        res = self.sess.run(fetches=fetches, feed_dict=feed_dict)

        # write summary for loss
        self.train_writer.add_summary(res['loss_summarization'], global_step=self.global_step)

        # write summary for images
        if self.global_step % 200 == 0:
            self.train_writer.add_summary(self.sess.run(self.summarization, feed_dict=feed_dict),
                                          global_step=self.global_step)

        self.global_step += 1

"""
References
----------
[1] Graves, Alex, Greg Wayne, and Ivo Danihelka. 
"Neural turing machines." arXiv preprint arXiv:1410.5401 (2014).
[2] https://github.com/carpedm20/NTM-tensorflow/blob/master/ops.py
"""
