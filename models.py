import random
import collections
import warnings
from six.moves import range
import numpy as np
import six
import tensorflow as tf
import tensorflow_federated as tff


ENCODER_SIZE = 256

def create_encoder_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            ENCODER_SIZE, activation=tf.nn.relu, input_shape=(784,), name='encoder1')
    ])
    return  model


def create_decoder_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
          784, activation=tf.nn.sigmoid, input_shape=(ENCODER_SIZE,), name='decoder1')
        ])
    return  model


def create_classifier_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
        10, activation=tf.nn.softmax, input_shape=(ENCODER_SIZE,), name='classifier1')
        ])
    return model


class Model(object):
    '''
    Your model must inherit from this class and specify a __call__ method which 
    returns a compiled tf model
    '''
    def __init__(self, opt):
        return

    def __call__(self):
        raise NotImplementedError('must define a class for your model that inherits \
                                  from ModelFunction and implement __call__ method. \
                                    The method must return a compiled keras model. \
                                  ')

    def create_tff_model_fn(self, sample_batch):
        keras_model = self()
        return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

    def save_model_weights(self, model_fp, federated_state):
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


class ClassifierModel(Model):
    def __init__(self, opt):
        self.opt = opt
        self.learning_rate = opt['learning_rate']
        self.saved_model_fp = opt['saved_model_fp']
        Model.__init__(self, opt)

    def __call__(self):
        '''
        Returns a compiled keras model.
        '''
        encoder_model = create_encoder_keras_model()
    
        if self.saved_model_fp:
            encoder_model.load_weights(self.saved_model_fp, by_name=True)
            
        model = tf.keras.models.Sequential([
            encoder_model,
            create_classifier_keras_model()
        ])
      
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model
    

class AutoencoderModel(Model):
    def __init__(self, opt):
        self.opt = opt
        self.learning_rate = opt['learning_rate']
        Model.__init__(self, opt)

    def __call__(self, saved_encoder_model=None):
        '''
        Returns a compiled keras model.
        '''
        encoder_model = create_encoder_keras_model()
    
        if saved_encoder_model:
            encoder_model.load_weights(saved_encoder_model)
            
        model = tf.keras.models.Sequential([
            encoder_model,
            create_decoder_keras_model()
        ])
      
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                        optimizer=tf.keras.optimizers.Adadelta(self.learning_rate, rho=0.95),
                        metrics=[tf.keras.metrics.MeanSquaredError()])
        return model
    
  