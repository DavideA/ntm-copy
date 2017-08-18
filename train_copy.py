import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.ion()

from NTM import NTM

sequence_length = 4
token_size = 10


def start_token():
    token = np.zeros(shape=(token_size,), dtype=np.float32)
    token[0] = 1
    return token


def end_token():
    token = np.zeros(shape=(token_size,), dtype=np.float32)
    token[1] = 1
    return token


def generate_copy_sample():

    sequence = []
    for _ in range(0, sequence_length):
        sequence.append(np.round(np.random.rand(token_size)))

    X = np.stack(sequence, axis=0)
    Y = X.copy()

    return start_token(), X, end_token(),  Y


def main():
    ntm = NTM(memory_capacity=64, memory_vector_size=128, input_vector_size=token_size, sequence_length=sequence_length)

    # print how many parameters are there
    all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    trainable_parameters = sum([np.prod([int(i) for i in v.get_shape()]) for v in all_variables])
    print('Total number of parameters: {:,}'.format(trainable_parameters))

    print('Training is starting, check your logs.')
    for _ in range(0, 999999):
        start, X, end, Y = generate_copy_sample()
        ntm.train_on_sample(start, X, end, Y)



if __name__ == '__main__':
    main()
