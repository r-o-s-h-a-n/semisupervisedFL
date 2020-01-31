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


def create_compiled_autoencoder_keras_model():
  model = tf.keras.models.Sequential([
      create_encoder_keras_model(),
      create_decoder_keras_model()
      ])
  
  model.compile(
      loss=tf.keras.losses.BinaryCrossentropy(),
      optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95),
      metrics=[])
  return model


def create_compiled_classifier_keras_model(encoder_model):  
  model = tf.keras.models.Sequential([
        encoder_model,
        create_classifier_keras_model()
      ])
  
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model


def autoencoder_model_fn(sample_autoencoder_batch):
  keras_model = create_compiled_autoencoder_keras_model()
  return tff.learning.from_compiled_keras_model(keras_model, sample_autoencoder_batch)


def classifier_model_fn(sample_classifier_batch, saved_encoder_model=None):
  encoder_model = create_encoder_keras_model()
  
  if saved_encoder_model:
      encoder_model.load_model(saved_encoder_model)
  
  keras_model = create_compiled_classifier_keras_model(encoder_model)
  return tff.learning.from_compiled_keras_model(keras_model, sample_classifier_batch)

