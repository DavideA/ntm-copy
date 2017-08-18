"""
This script is for training the Neural Turing Machine for 
the copy task.
"""

import numpy as np
import argparse

from NTM import NTM


def start_token(token_size):
    """
    Provides a start token.
    
    Parameters
    ----------
    token_size: int
        the size of the token.

    Returns
    -------
    token: ndarray
        the start token having shape (token_size,).
    """

    token = np.zeros(shape=(token_size,), dtype=np.float32)
    token[0] = 1
    return token


def end_token(token_size):
    """
    Provides an end token.

    Parameters
    ----------
    token_size: int
        the size of the token.

    Returns
    -------
    token: ndarray
        the end token having shape (token_size,).
    """

    token = np.zeros(shape=(token_size,), dtype=np.float32)
    token[1] = 1
    return token


def generate_copy_sample(args):
    """
    Provides an input-target example for training the NTM.
    
    Parameters
    ----------
    args: 
        the command line arguments.

    Returns
    -------
    tuple
        a tuple like (start_token, input, end_token, targets).
    """

    sequence = []
    for _ in range(0, args.sequence_length):
        sequence.append(np.round(np.random.rand(args.token_size)))

    X = np.stack(sequence, axis=0)
    Y = X.copy()

    return start_token(args.token_size), X, end_token(args.token_size),  Y


def parse_arguments():
    """
    Parses command line arguments.
    
    Returns
    -------
    args:
        the command line arguments.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sequence_length', type=int, default=3, help='The length of the sequence to copy', metavar='')
    parser.add_argument('--token_size', type=int, default=10,
                        help='The size of the tokens making the sequence', metavar='')
    parser.add_argument('--memory_capacity', type=int, default=64,
                        help='Number of records that can be stored in memory', metavar='')
    parser.add_argument('--memory_vector_size', type=int, default=128,
                        help='Dimensionality of records stored in memory', metavar='')
    parser.add_argument('--training_samples', type=int, default=999999,
                        help='Number of training samples', metavar='')
    parser.add_argument('--controller_output_dim', type=int, default=256,
                        help='Dimensionality of the feature vector produced by the controller', metavar='')
    parser.add_argument('--controller_hidden_dim', type=int, default=512,
                        help='Dimensionality of the hidden layer of the controller', metavar='')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Optimizer learning rate', metavar='')
    parser.add_argument('--min_grad', type=float, default=-10.,
                        help='Minimum value of gradient clipping', metavar='')
    parser.add_argument('--max_grad', type=float, default=10.,
                        help='Maximum value of gradient clipping', metavar='')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='The directory where to store logs', metavar='')

    return parser.parse_args()


def main():
    """ Main function """

    # Parse command line arguments
    args = parse_arguments()

    # Build model
    ntm = NTM(memory_capacity=args.memory_capacity,
              memory_vector_size=args.memory_vector_size,
              input_vector_size=args.token_size,
              sequence_length=args.sequence_length,
              controller_out_dim=args.controller_output_dim,
              controller_hidden_dim=args.controller_hidden_dim,
              learning_rate=args.learning_rate,
              min_grad=args.min_grad,
              max_grad=args.max_grad,
              logdir=args.logdir)

    # train model
    print('Training is starting, check your logs.')
    for _ in range(0, args.training_samples):
        start, X, end, Y = generate_copy_sample(args)
        ntm.train_on_sample(start, X, end, Y)


# entry point
if __name__ == '__main__':
    main()
