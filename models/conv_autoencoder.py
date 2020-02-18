import random
import collections
import warnings
from six.moves import range
import numpy as np
import six
import tensorflow as tf
import tensorflow_federated as tff
from models.model import Model


INPUT_SHAPES = {'emnist': (28,28,1), 'cifar100': (32,32,3), 'cifar10central': (32,32,3)}
OUTPUT_SHAPES = {'emnist': 10, 'cifar100': 100, 'cifar10central': 10}

def create_encoder_keras_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape, name='encoder1'),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='encoder2'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
    ])
    return  model

def create_decoder_keras_model(output_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((output_shape[0],output_shape[1],64)),
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='decoder1'),
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='decoder2'),
        tf.keras.layers.Conv2DTranspose(filters=output_shape[-1], kernel_size=(1, 1), name='decoder3')
        ])
    return  model

def create_classifier_keras_model(output_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', name='classifier1'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(output_shape, activation=tf.nn.softmax, name='classifier2')
        ])
    return model


class ConvSupervisedModel(Model):
    def __init__(self, ph):
        Model.__init__(self, ph)
        self.input_shape = INPUT_SHAPES[self.ph['dataset']]
        self.output_shape = OUTPUT_SHAPES[self.ph['dataset']]
        self.pretrained_model_fp = self.ph.setdefault('pretrained_model_fp', None)

    def __call__(self):
        '''
        Returns a compiled keras model.
        '''
        encoder_model = create_encoder_keras_model(self.input_shape)
        
        if self.pretrained_model_fp:
            encoder_model.load_weights(self.pretrained_model_fp, by_name=True)

        model = tf.keras.models.Sequential([
            encoder_model,
            create_classifier_keras_model(self.output_shape)
        ])
      
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=self.optimizer(learning_rate=self.learning_rate,
                                    nesterov = self.nesterov,
                                    momentum = self.momentum,
                                    decay = self.decay
                    ),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model

    def preprocess_emnist(self,
                    dataset, 
                    num_epochs, 
                    shuffle_buffer, 
                    batch_size):

        def element_fn(element):
            img = tf.expand_dims(element['pixels'], 2)

            return (img,
                    tf.reshape(element['label'], [1]))

        return dataset.filter(lambda x: not x['is_masked_supervised'] if 'is_masked_supervised' in x else True).repeat(
            num_epochs).map(element_fn).shuffle(shuffle_buffer).batch(batch_size)

    def preprocess_cifar100(self,
                    dataset,
                    num_epochs, 
                    shuffle_buffer, 
                    batch_size):

        def element_fn(element):
            img = tf.math.divide(tf.cast(element['image'], tf.float32),
                                tf.constant(255.0, dtype=tf.float32))

            return (img,
                tf.reshape(element['label'], [1]))

        return dataset.filter(lambda x: not x['is_masked_unsupervised'] if 'is_masked_unsupervised' in x else True).repeat(
            num_epochs).map(element_fn).shuffle(shuffle_buffer).batch(batch_size)

    def preprocess_cifar10central(self,
                    dataset,
                    num_epochs, 
                    shuffle_buffer, 
                    batch_size):

        def element_fn(element):
            img = tf.math.divide(tf.cast(element['image'], tf.float32),
                                tf.constant(255.0, dtype=tf.float32))

            return (img,
                tf.reshape(element['label'], [1]))

        return dataset.filter(lambda x: not x['is_masked_unsupervised'] if 'is_masked_unsupervised' in x else True).repeat(
            num_epochs).map(element_fn).shuffle(shuffle_buffer).batch(batch_size)


class ConvAutoencoderModel(Model):
    def __init__(self, ph):
        Model.__init__(self, ph)
        self.input_shape = INPUT_SHAPES[self.ph['dataset']]

    def __call__(self):
        '''
        Returns a compiled keras model.
        '''
        model = tf.keras.models.Sequential([
            create_encoder_keras_model(self.input_shape),
            create_decoder_keras_model(self.input_shape)
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
                    batch_size):

        def element_fn(element):
            img = tf.expand_dims(element['pixels'], 2)

            return (img,
                    img)

        return dataset.filter(lambda x: not x['is_masked_supervised'] if 'is_masked_supervised' in x else True).repeat(
            num_epochs).map(element_fn).shuffle(shuffle_buffer).batch(batch_size)
    
    def preprocess_cifar100(self,
                    dataset,
                    num_epochs, 
                    shuffle_buffer, 
                    batch_size):

        def element_fn(element):
            img = tf.math.divide(tf.cast(element['image'], tf.float32),
                    tf.constant(255.0, dtype=tf.float32))

            return (img,
                    img)

        return dataset.filter(lambda x: not x['is_masked_unsupervised'] if 'is_masked_unsupervised' in x else True).repeat(
            num_epochs).map(element_fn).shuffle(shuffle_buffer).batch(batch_size)

    def preprocess_cifar10central(self,
                    dataset,
                    num_epochs, 
                    shuffle_buffer, 
                    batch_size):

        def element_fn(element):
            img = tf.math.divide(tf.cast(element['image'], tf.float32),
                    tf.constant(255.0, dtype=tf.float32))

            return (img,
                    img)

        return dataset.filter(lambda x: not x['is_masked_unsupervised'] if 'is_masked_unsupervised' in x else True).repeat(
            num_epochs).map(element_fn).shuffle(shuffle_buffer).batch(batch_size)
