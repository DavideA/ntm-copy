# ntm-copy
Yet another Tensorflow Neural Turing Machine.

### Why?
I implemented it from scratch after reading the paper 
just for the sake of study and coding practise. 
Having a terrible time training it, I peeked into
[this famous repo](https://github.com/carpedm20/NTM-tensorflow) to
get some insights every once in a while.

### Naive
This is as simple as NTMs can get:
* it is hard-coded for the copy task.
* features a 2-layer feed-forward controller 
(as opposed to a LSTM one).
* has a single read head and a single write head.
* processes fixed length sequence.

...yet it's good for studying.

---
More docs soon...
