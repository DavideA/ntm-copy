# ntm-copy
Yet another Tensorflow Neural Turing Machine.

### Why?
I implemented it from scratch after reading the paper 
just for the sake of study and coding practice. 
Having a terrible time training it, I peeked into
[this famous repo](https://github.com/carpedm20/NTM-tensorflow) to
get some insights every once in a while.

### Naive
This is as simple as NTMs can get:
* it is hard-coded for the copy task.
* features a 2-layer feed-forward controller 
(as opposed to a LSTM one).
* has a single read head and a single write head.
* processes fixed length sequences.

...yet it's good for studying.

---
### How to use
```
usage: train_copy.py [-h] [--sequence_length] [--token_size]
                     [--memory_capacity] [--memory_vector_size]
                     [--training_samples] [--controller_output_dim]
                     [--controller_hidden_dim] [--learning_rate] [--min_grad]
                     [--max_grad] [--logdir]

optional arguments:
  -h, --help              show this help message and exit
  --sequence_length       The length of the sequence to copy (default: 3)
  --token_size            The size of the tokens making the sequence (default: 10)
  --memory_capacity       Number of records that can be stored in memory (default: 64)
  --memory_vector_size    Dimensionality of records stored in memory (default: 128)
  --training_samples      Number of training samples (default: 999999)
  --controller_output_dim Dimensionality of the feature vector produced by the controller (default: 256)
  --controller_hidden_dim Dimensionality of the hidden layer of the controller (default: 512)
  --learning_rate         Optimizer learning rate (default: 0.0001)
  --min_grad              Minimum value of gradient clipping (default: -10.0)
  --max_grad              Maximum value of gradient clipping (default: 10.0)
  --logdir                The directory where to store logs (default: logs)
```

During learning, you can monitor loss and other stuff with tensorboard:
##### loss
![alt text](/img/loss "Loss value")
