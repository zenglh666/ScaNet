from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class DenseModel(object):
    """ Abstract object representing an NMT model """

    def __init__(self, params, scope):
        self._scope = scope
        self._params = params

    def model_func(self, images, lables, mode, initializer=None, regularizer=None):
        """
        :param initializer: the initializer used to initialize the model
        :param regularizer: the regularizer used for model regularization
        """
        raise NotImplementedError("Not implemented")

    @staticmethod
    def get_name():
        raise NotImplementedError("Not implemented")

    @staticmethod
    def get_parameters():
        raise NotImplementedError("Not implemented")

    @property
    def parameters(self):
        return self._params
