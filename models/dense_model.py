import random
import collections
import warnings
from six.moves import range
import numpy as np
import six
import tensorflow as tf
import tensorflow_federated as tff
from models.model import Model


ENCODER_SIZE = 256
INPUT_SIZE = {'emnist': 784, 'cifar100': 3072, 'cifar10central': 3072}
OUTPUT_SIZE = {'emnist': 10, 'cifar100': 20, 'cifar10central': 10}


def create_encoder_keras_model(input_size, trainable=True):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            ENCODER_SIZE, activation=tf.nn.relu, input_shape=(input_size,), name='encoder1', trainable=trainable)
    ], name='encoder')
    return  model


def create_decoder_keras_model(output_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
          output_size, activation=tf.nn.sigmoid, input_shape=(ENCODER_SIZE,), name='decoder1')
        ], name='decoder')
    return  model


def create_classifier_keras_model(output_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
        output_size, activation=tf.nn.softmax, input_shape=(ENCODER_SIZE,), name='classifier1')
        ], name='classifier')
    return model


class DenseSupervisedModel(Model):
    def __init__(self, ph):
        Model.__init__(self, ph)
        self.input_size = INPUT_SIZE[self.ph['dataset']]
        self.output_size = OUTPUT_SIZE[self.ph['dataset']]
        self.pretrained_model_fp = self.ph.setdefault('pretrained_model_fp', None)
        self.fine_tune = self.ph['fine_tune']

        if self.pretrained_model_fp:
            print('training on pretrained model')
        else:
            print('training from scratch without pretrained model')

        if self.fine_tune:
            print('fine tuning pretrained model')
        else:
            print('pretrained model weights are frozen')

    def __call__(self):
        '''
        Returns a compiled keras model.
        '''
        encoder_model = create_encoder_keras_model(self.input_size, trainable=self.fine_tune)

        model = tf.keras.models.Sequential([
            encoder_model,
            create_classifier_keras_model(self.output_size)
        ])
      
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=self.optimizer(learning_rate=self.learning_rate,
                                        nesterov=self.nesterov,
                                        momentum=self.momentum, 
                                        decay=self.decay),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        
        if self.pretrained_model_fp:
            model.load_weights(self.pretrained_model_fp, by_name=True)

        return model

    def preprocess_emnist(self,
                    dataset, 
                    num_epochs, 
                    shuffle_buffer, 
                    batch_size,
                    learning_env):

        assert learning_env in ('central', 'federated')
        if learning_env == 'central':
            num_epochs = 1

        def element_fn(element):
            return (tf.reshape(element['pixels'], [-1]),
                    tf.reshape(element['label'], [1]))

        return dataset.filter(lambda x: not x['is_masked_supervised'] if 'is_masked_supervised' in x else True).repeat(
            num_epochs).map(element_fn).shuffle(shuffle_buffer).batch(batch_size)

    def preprocess_cifar100(self,
                    dataset,
                    num_epochs, 
                    shuffle_buffer, 
                    batch_size,
                    learning_env):

        assert learning_env in ('central', 'federated')
        if learning_env == 'central':
            num_epochs = 1

        def element_fn(element):
            img = tf.math.divide(tf.cast(element['image'], tf.float32),
                                tf.constant(255.0, dtype=tf.float32))

            return (tf.reshape(img, [-1]),
                tf.reshape(element['coarse_label'], [1]))

        return dataset.filter(lambda x: not x['is_masked_unsupervised'] if 'is_masked_unsupervised' in x else True).repeat(
            num_epochs).map(element_fn).shuffle(shuffle_buffer).batch(batch_size)

    def preprocess_cifar10central(self,
                    dataset,
                    num_epochs, 
                    shuffle_buffer, 
                    batch_size,
                    learning_env):

        assert learning_env in ('central', 'federated')
        if learning_env == 'central':
            num_epochs = 1

        def element_fn(element):
            img = tf.math.divide(tf.cast(element['image'], tf.float32),
                                tf.constant(255.0, dtype=tf.float32))

            return (tf.reshape(img, [-1]),
                tf.reshape(element['label'], [1]))

        return dataset.filter(lambda x: not x['is_masked_unsupervised'] if 'is_masked_unsupervised' in x else True).repeat(
            num_epochs).map(element_fn).shuffle(shuffle_buffer).batch(batch_size)


class DenseAutoencoderModel(Model):
    def __init__(self, ph):
        Model.__init__(self, ph)
        self.input_size = INPUT_SIZE[self.ph['dataset']]

    def __call__(self):
        '''
        Returns a compiled keras model.
        '''
        encoder_model = create_encoder_keras_model(self.input_size)
    
        model = tf.keras.models.Sequential([
            encoder_model,
            create_decoder_keras_model(self.input_size)
        ])
      
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                        optimizer=self.optimizer(learning_rate=self.learning_rate,
                                                nesterov=self.nesterov,
                                                momentum=self.momentum, 
                                                decay=self.decay),
                        metrics=[tf.keras.metrics.MeanSquaredError()])
        return model
    
    def preprocess_emnist(self,
                    dataset,
                    num_epochs, 
                    shuffle_buffer, 
                    batch_size,
                    learning_env):

        assert learning_env in ('central', 'federated')
        if learning_env == 'central':
            num_epochs = 1

        def element_fn(element):
            return (tf.reshape(element['pixels'], [-1]),
                tf.reshape(element['pixels'], [-1]))

        return dataset.filter(lambda x: not x['is_masked_unsupervised'] if 'is_masked_unsupervised' in x else True).repeat(
            num_epochs).map(element_fn).shuffle(shuffle_buffer).batch(batch_size)

    def preprocess_cifar100(self,
                    dataset,
                    num_epochs, 
                    shuffle_buffer, 
                    batch_size,
                    learning_env):

        assert learning_env in ('central', 'federated')
        if learning_env == 'central':
            num_epochs = 1

        def element_fn(element):
            img = tf.math.divide(tf.cast(element['image'], tf.float32),
                    tf.constant(255.0, dtype=tf.float32))

            return (tf.reshape(img, [-1]),
                tf.reshape(img, [-1]))

        return dataset.filter(lambda x: not x['is_masked_unsupervised'] if 'is_masked_unsupervised' in x else True).repeat(
            num_epochs).map(element_fn).shuffle(shuffle_buffer).batch(batch_size)

    def preprocess_cifar10central(self,
                    dataset,
                    num_epochs, 
                    shuffle_buffer, 
                    batch_size,
                    learning_env):

        assert learning_env in ('central', 'federated')
        if learning_env == 'central':
            num_epochs = 1

        def element_fn(element):
            img = tf.math.divide(tf.cast(element['image'], tf.float32),
                    tf.constant(255.0, dtype=tf.float32))

            return (tf.reshape(img, [-1]),
                tf.reshape(img, [-1]))

        return dataset.filter(lambda x: not x['is_masked_unsupervised'] if 'is_masked_unsupervised' in x else True).repeat(
            num_epochs).map(element_fn).shuffle(shuffle_buffer).batch(batch_size)
