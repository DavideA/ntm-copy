import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.ion()

from NTM import NTM

sequence_length = 3
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

    f, axarr = plt.subplots(3, 3)
    for _ in xrange(0, 999999):
        start, X, end, Y = generate_copy_sample()
        loss, a_t, ww_t, e_t, rw_t, r_t, out, memory = ntm.train_on_sample(start, X, end, Y)

        if _ % 300 == 0:

            # visualize
            axarr[0, 0].imshow(X[:sequence_length+2].T, cmap='jet')
            axarr[0, 0].set_title('inputs')
            axarr[0, 1].imshow(Y[-sequence_length:].T, cmap='jet')
            axarr[0, 1].set_title('targets')
            axarr[0, 2].imshow(out[-sequence_length:].T, cmap='jet')
            axarr[0, 2].set_title('outputs')
            axarr[1, 0].imshow(ww_t.T, cmap='jet')
            axarr[1, 0].set_title('write locations')
            axarr[1, 1].imshow(e_t.T, cmap='jet')
            axarr[1, 1].set_title('erase vectors')
            axarr[1, 2].imshow(a_t.T, cmap='jet')
            axarr[1, 2].set_title('add vectors')
            axarr[2, 0].imshow(rw_t.T, cmap='jet')
            axarr[2, 0].set_title('read locations')
            axarr[2, 1].imshow(r_t.T, cmap='jet')
            axarr[2, 1].set_title('read vectors')
            axarr[2, 2].imshow(memory[-1], cmap='jet')
            axarr[2, 2].set_title('memory')

            # f, axarr = plt.subplots(2*sequence_length+2)
            # for i, m in enumerate(memory):
            #     axarr[i].imshow(m)

            plt.show(block=False)
            plt.pause(0.02)

            if _ % 1000 == 0:
                plt.savefig('{:03d}.png'.format(_))
        print(loss)


if __name__ == '__main__':
    main()
