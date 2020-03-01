import random
import collections
import warnings
from six.moves import range
import numpy as np
import six
import tensorflow as tf
import tensorflow_federated as tff


class Model(object):
    '''
    Your model must inherit from this class and specify: 
        1. a __call__ method which returns a compiled tf model
        2. a preprocess method which preprocess a dataset for use in the model
    '''
    def __init__(self, ph):
        self.ph=ph
        self.optimizer = getattr(tf.keras.optimizers, ph['optimizer'])
        self.learning_rate = ph['learning_rate']
        self.nesterov = ph.setdefault('nesterov', False)
        self.momentum = ph.setdefault('momentum', 0.0)
        self.decay = ph.setdefault('decay', 0.0)

    def __call__(self):
        raise NotImplementedError('must define a class for your model that inherits \
                                  from ModelFunction and implement __call__ method. \
                                    The method must return a compiled keras model. \
                                  ')

    def create_tff_model_fn(self, sample_batch):
        keras_model = self()
        return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

    def save_model_weights(self, model_fp, federated_state, sample_batch):
        '''
        Saves model weights to file from tff iteration state.

        Arguments:
            model_fp: str,      file path to store model weights
            federated_state: tff.python.common_libs.anonymous_tuple.AnonymousTuple, 
                the federated training state from which you wish to save the model weights.

        Returns:
            nothing, but saves model weights
        '''
        keras_model = self()
        keras_model.build(sample_batch)
        tff.learning.assign_weights_to_keras_model(keras_model, federated_state.model)
        keras_model.save_weights(model_fp)
        return

    def load_model_weights(self, model_fp):
        '''
        Loads the model weights to a new model object using create_model_fn.
        
        Arguments:
            model_fp: str,      file path where model weights are stored.

        Returns:
            model: tf.keras.Model, a keras model with weights initialized
        '''
        keras_model = self()
        keras_model.load_weights(model_fp)
        return keras_model