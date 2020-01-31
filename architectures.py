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
          ENCODER_SIZE, activation=tf.nn.relu, input_shape=(784,))
  ])
  return  model


def create_decoder_keras_model():
  model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
          784, activation=tf.nn.sigmoid, input_shape=(ENCODER_SIZE,))
      ])
  return  model


def create_classifier_keras_model():
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(
        10, activation=tf.nn.softmax, input_shape=(ENCODER_SIZE,))
    ])
  return model


def create_compiled_autoencoder_keras_model(opt):
  model = tf.keras.models.Sequential([
      create_encoder_keras_model(),
      create_decoder_keras_model()
      ])
  
  model.compile(
      loss=tf.keras.losses.BinaryCrossentropy(),
      optimizer=tf.keras.optimizers.Adadelta(learning_rate=opt['learning_rate'], rho=0.95), #1.0
      metrics=[])
  return model


def create_compiled_classifier_keras_model(opt):
    encoder_model = create_encoder_keras_model()
    if opt['saved_model_fp']:
        encoder_model.load_model(opt['saved_model_fp'])
      
    model = tf.keras.models.Sequential([
          encoder_model,
          create_classifier_keras_model()
        ])
    
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=opt['learning_rate']), #0.02
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


def create_tff_model_fn(create_complied_keras_model_fn, sample_batch):
  keras_model = create_complied_keras_model_fn()
  return tff.learning.from_compiled_keras_model(keras_model, sample_batch)


class Model(object):
  '''
  Your model must inherit from this and define a method `self.create_complied_keras_model_fn`
  which returns a compiled keras model
  '''
  def __init__(self):
    return

  def create_tff_model_fn(create_complied_keras_model_fn, sample_batch):
    keras_model = create_complied_keras_model_fn()
    return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

  def save_model_weights(model_fp, create_model_fn, federated_state):
      '''
      Saves model weights to file from tff iteration state.

      Arguments:
          model_fp: str, filepath to save the model.
          create_model_fn: fnc, function that creates the keras model.
          federated_state: tff.python.common_libs.anonymous_tuple.AnonymousTuple, 
              the federated training state from which you wish to save the model weights.

      Returns:
          nothing, but saves model weights to disk
      '''
      model = create_model_fn()
      print(type(model))
      tff.learning.assign_weights_to_keras_model(model, federated_state.model)
      model.save_weights(model_fp)
      return


  def load_model_weights(model_fp, create_model_fn):
      '''
      Loads the model weights to a new model object using create_model_fn.
      
      Arguments:
          model_fp: String, filepath to the saved model weights.
          create_model_fn: fnc, function that creates the keras model.

      Returns:
          model: tf.keras.Model, a keras model with weights initialized
      '''
      return create_model_fn().load_model(model_fp)
