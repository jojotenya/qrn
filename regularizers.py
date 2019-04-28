# Copyright (c) 2017 Ben Poole & Friedemann Zenke
# MIT License -- see LICENSE for details
# 
# This file is part of the code to reproduce the core results of:
# Zenke, F., Poole, B., and Ganguli, S. (2017). Continual Learning Through
# Synaptic Intelligence. In Proceedings of the 34th International Conference on
# Machine Learning, D. Precup, and Y.W. Teh, eds. (International Convention
# Centre, Sydney, Australia: PMLR), pp. 3987-3995.
# http://proceedings.mlr.press/v70/zenke17a.html
#
import tensorflow as tf

def quadratic_regularizer(weights, tensors, norm=2):
    """Compute the regularization term.

    Args:
        weights: list of Variables
        _tensors: dict from variable name to dictionary containing the variables.
            Each set of variables is stored as a dictionary mapping from weights to variables.
            For example, tensors['grads'][w] would retreive the 'grads' variable for weight w
        norm: power for the norm of the (weights - consolidated weight)

    Returns:
        scalar Tensor regularization term
    """
    reg = 0.0
    for w in range(len(weights)):
        reg += tf.reduce_sum(tensors['omega'][w] * (w - tensors['star_vars'][w])**norm)
    return reg

def get_power_regularizer(power=2.0):
    """Power regularizers with different norms"""
    def _regularizer_fn(weights, tensors):
        return quadratic_regularizer(weights, tensors, norm=power)
    return _regularizer_fn
