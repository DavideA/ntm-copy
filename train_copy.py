import numpy as np
import matplotlib.pyplot as plt

from NTM import NTM

sequence_length = 10
token_size = 10


def initial_token():
    return np.zeros(shape=(token_size,), dtype=np.float32)


def last_token():
    token = np.zeros(shape=(token_size,), dtype=np.float32)
    token[0] = 1
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
    ntm = NTM(memory_capacity=512, memory_vector_size=128, input_vector_size=token_size)

    while True:
        ntm.train_on_sample(*generate_copy_sample())
        ntm.reset()


if __name__ == '__main__':
    main()
