import random
import collections
import warnings
from six.moves import range
import numpy as np
import six
import tensorflow as tf
import tensorflow_federated as tff
from models.model import Model


class ConvDropoutSupervisedModel(Model):
    def __init__(self, ph):
        Model.__init__(self, ph)

    def __call__(self):
        '''
        Returns a compiled keras model.
        '''
        data_format = 'channels_last'
        input_shape = (784,)
    
        # if self.saved_model_fp:
        #     encoder_model.load_weights(self.saved_model_fp, by_name=True)
            
        # model = tf.keras.models.Sequential([
        #     encoder_model,
        #     create_classifier_keras_model()
        # ])

        model = tf.keras.models.Sequential([
          tf.keras.layers.Conv2D(
              32,
              kernel_size=(3, 3),
              activation='relu',
              input_shape=input_shape,
              data_format=data_format),
          tf.keras.layers.Conv2D(
              64, kernel_size=(3, 3), activation='relu', data_format=data_format),
          tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
          tf.keras.layers.Dropout(0.25),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Dense(
              10 if only_digits else 62, activation=tf.nn.softmax),
        ])
      
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=self.optimizer(learning_rate=self.learning_rate),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model

    def preprocess(self,
                    dataset, 
                    num_epochs, 
                    shuffle_buffer, 
                    batch_size):

        def element_fn(element):
            return (tf.reshape(element['pixels'], [-1]),
                    tf.reshape(element['label'], [1]))

        return dataset.filter(lambda x: not x['is_masked_supervised'] if 'is_masked_supervised' in x else True).repeat(
            num_epochs).map(element_fn).shuffle(shuffle_buffer).batch(batch_size)


