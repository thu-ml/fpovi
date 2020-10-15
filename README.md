This branch contains a TensorFlow 2 implementation of f-POVI on a LSTM+attention model. 
It is hopefully a more clear demonstration for implementing the algorithm on imperative 
DL frameworks.

For the fPOVI implementation see `trainers.fPOVITrainer`. Compare it with the other 
trainers, which implements (ensembled versions) of SGD, mean-field VI and recurrent 
dropout.

We use the sampling-based prior approximation as described in the paper. This (along 
with the need to implement other baselines) is the reason we need to fork the NN library
from sonnet (to `modules.py`). For prototyping we recommend to drop the prior term and
rely on arly stopping for regularization, since you can use the standard DNN libraries
then. The performance drop will be noticeable but you should still be able to get some 
meaningful uncertainty estimates. For more accurate priors we recommend to use the
function-space prior corresponding to an infinite-width BNN with similar architectures.
