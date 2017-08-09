import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.ion()

from NTM import NTM

sequence_length = 10
token_size = 8


def initial_token():
    token = np.zeros(shape=(token_size,), dtype=np.float32)
    token[0] = 1
    return token


def last_token():
    token = np.zeros(shape=(token_size,), dtype=np.float32)
    token[1] = 1
    return token


def generate_copy_sample():

    sequence = []
    for _ in range(0, sequence_length):
        sequence.append(np.round(np.random.rand(token_size)))

    X = np.stack([initial_token()] + sequence + [last_token()] +
                 [np.zeros(shape=(token_size,), dtype=np.float32)] * sequence_length,
                 axis=0)
    Y = np.stack([np.zeros(shape=(token_size,), dtype=np.float32)] * (sequence_length + 2) + sequence,
                 axis=0)

    if False:
        f, axarr = plt.subplots(2)
        axarr[0].imshow(X.T, cmap='jet')
        axarr[1].imshow(Y.T, cmap='jet')
        plt.show()

    return X, Y


def main():
    ntm = NTM(memory_capacity=128, memory_vector_size=64, input_vector_size=token_size, sequence_length=2*sequence_length+2)
    all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    trainable_parameters = sum([np.prod([int(i) for i in v.get_shape()]) for v in all_variables])
    print('Total number of parameters: {:,}'.format(trainable_parameters))

    plt.clf()
    f, axarr = plt.subplots(3, 3)
    for _ in xrange(0, 999999):
        X, Y = generate_copy_sample()
        loss, a_t, ww_t, e_t, rw_t, r_t, out, memory = ntm.train_on_sample(X, Y)

        # visualize
        axarr[0, 0].imshow(X.T, cmap='jet')
        axarr[0, 0].set_title('inputs')
        axarr[0, 1].imshow(Y.T, cmap='jet')
        axarr[0, 1].set_title('targets')
        axarr[0, 2].imshow(out.T, cmap='jet')
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

        if _ % 300 == 0:
            plt.savefig('{:03d}.pdf'.format(_))
        print(loss)

        ntm.reset()


if __name__ == '__main__':
    main()
