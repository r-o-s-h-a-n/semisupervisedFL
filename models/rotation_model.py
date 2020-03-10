import random
import collections
import warnings
from six.moves import range
import numpy as np
import math
import six
import tensorflow as tf
import tensorflow_federated as tff

from models.model import Model
from models.initializers import ConvInitializer, DenseInitializer
from models.layers import create_NIN_block, GlobalAveragePooling


'''
"Deep model" refers to our implementation of the full NIN rotation net model described in 
            Gidaris, Spyros, Praveer Singh, and Nikos Komodakis. "Unsupervised representation 
            learning by predicting image rotations." arXiv preprint arXiv:1803.07728 (2018).

"Simple model" refers to a shallower network used in our experiments which is based on the 
            deep model.
'''


DEEP_NCHANNELS1 = 192
DEEP_NCHANNELS2 = 160
DEEP_NCHANNELS3 = 96

SIMPLE_NCHANNELS1 = 32
SIMPLE_NCHANNELS2 = 64

INPUT_SHAPES = {'emnist': [28,28,1], 'cifar100': [32,32,3], 'cifar10central': [32,32,3]}
OUTPUT_SHAPES = {'emnist': 10, 'cifar100': 20, 'cifar10central': 10}


def create_deep_feature_extractor_block(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape),
        # block 1
        create_NIN_block(DEEP_NCHANNELS1, 5, name='F_Block1_Conv1', input_shape=input_shape),
        create_NIN_block(DEEP_NCHANNELS2, 1, name='F_Block1_Conv2'),
        create_NIN_block(DEEP_NCHANNELS3, 1, name='F_Block1_Conv3'),
        tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='same', name='F_Block1_MaxPool'),

        # block 2
        create_NIN_block(DEEP_NCHANNELS1, 5, name='F_Block2_Conv1'),
        create_NIN_block(DEEP_NCHANNELS1, 1, name='F_Block2_Conv2'),
        create_NIN_block(DEEP_NCHANNELS1, 1, name='F_Block2_Conv3'),
        tf.keras.layers.AveragePooling2D(pool_size=3,strides=2,padding='same', name='F_Block2_AvgPool')
    ], name='Feature_Extractor')

    return model


def create_deep_label_classifier_block(input_shape, num_classes=10):
    model = tf.keras.models.Sequential([
        # block 3
        create_NIN_block(DEEP_NCHANNELS1, 3, name='L_Block3_Conv1'),
        create_NIN_block(DEEP_NCHANNELS1, 1, name='L_Block3_Conv2'),
        create_NIN_block(DEEP_NCHANNELS1, 1, name='L_Block3_Conv3'),

        GlobalAveragePooling(name='L_Global_Avg_Pool'),
        tf.keras.layers.Dense(num_classes, 
                                name='L_Linear_Classifier', 
                                activation='softmax',
                                kernel_initializer=DenseInitializer(num_classes))
    ],
    name = 'Label_Classifier')
    return model


def create_deep_rotation_classifier_block(input_shape, num_classes=4):
    model = tf.keras.models.Sequential([
        # block 3
        create_NIN_block(DEEP_NCHANNELS1, 3, name='R_Block3_Conv1'),
        create_NIN_block(DEEP_NCHANNELS1, 1, name='R_Block3_Conv2'),
        create_NIN_block(DEEP_NCHANNELS1, 1, name='R_Block3_Conv3'),

        # block 4
        create_NIN_block(DEEP_NCHANNELS1, 3, name='R_Block4_Conv1'),
        create_NIN_block(DEEP_NCHANNELS1, 1, name='R_Block4_Conv2'),
        create_NIN_block(DEEP_NCHANNELS1, 1, name='R_Block4_Conv3'),

        GlobalAveragePooling(name='R_Global_Avg_Pool'),
        tf.keras.layers.Dense(num_classes,
                                name='R_Linear_Classifier', 
                                activation='softmax',
                                kernel_initializer=DenseInitializer(num_classes))
    ],
    name = 'Rotation_Classifier')
    return model


def create_simple_feature_extractor_block(input_shape):
    model = tf.keras.models.Sequential([
        # block 1
        create_NIN_block(SIMPLE_NCHANNELS1, 3, name='F_Block1_Conv1', input_shape=input_shape),
        create_NIN_block(SIMPLE_NCHANNELS2, 1, name='F_Block1_Conv2'),
        create_NIN_block(SIMPLE_NCHANNELS2, 1, name='F_Block1_Conv3'),

        tf.keras.layers.MaxPooling2D(3, strides=2, padding='same', name='F_maxpool')
    ], name='Feature_Extractor')

    return model


def create_simple_label_classifier_block(input_shape, num_classes=10):
    model = tf.keras.models.Sequential([
        # block 2
        create_NIN_block(SIMPLE_NCHANNELS2, 3, name='L_Block2_Conv1'),
        create_NIN_block(SIMPLE_NCHANNELS2, 1, name='L_Block2_Conv2'),
        create_NIN_block(SIMPLE_NCHANNELS2, 1, name='L_Block2_Conv3'),

        tf.keras.layers.MaxPooling2D(3, strides=2, padding='same', name='L_maxpool'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, 
                                name='L_Hidden_Layer', 
                                activation='relu'
                                ),

        tf.keras.layers.Dense(num_classes, 
                                name='L_Linear_Classifier', 
                                activation='softmax'
                                )
    ],
    name = 'Label_Classifier')
    return model


def create_simple_rotation_classifier_block(input_shape, num_classes=4):
    model = tf.keras.models.Sequential([
        # block 2
        create_NIN_block(SIMPLE_NCHANNELS1, 5, name='R_Block2_Conv1'),
        create_NIN_block(SIMPLE_NCHANNELS2, 1, name='R_Block2_Conv2'),
        create_NIN_block(SIMPLE_NCHANNELS2, 1, name='R_Block2_Conv3'),

        tf.keras.layers.MaxPooling2D(3, strides=2, padding='same', name='L_maxpool'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, 
                                name='R_Hidden_Layer', 
                                activation='relu',
                                ),
        tf.keras.layers.Dense(num_classes, 
                                name='R_Linear_Classifier', 
                                activation='softmax',
                                )
    ],
    name = 'Rotation_Classifier')
    return model


class RotationSupervisedModel(Model):
    def __init__(self, ph):
        Model.__init__(self, ph)
        self.input_shape = INPUT_SHAPES[ph['dataset']]
        self.output_shape = OUTPUT_SHAPES[self.ph['dataset']]
        self.pretrained_model_fp = self.ph.setdefault('pretrained_model_fp', None)

        if self.pretrained_model_fp:
            print('training on a pretrained model')

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
            return (tf.expand_dims(element['pixels'], 2),
                    tf.reshape(element['label'], [1]))

        return dataset.filter(lambda x: not x['is_masked_supervised'] if 'is_masked_supervised' in x else True).map(element_fn
            ).repeat(num_epochs).shuffle(shuffle_buffer).batch(batch_size)

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
            img = element['image']
            img = tf.cast(img, tf.float32)
            img = tf.math.subtract(img, tf.convert_to_tensor([255*0.49139968, 255*0.48215841, 255*0.44653091], dtype=tf.float32))
            img = tf.math.divide(img, tf.convert_to_tensor([255*0.24703223, 255*0.24348513, 255*0.26158784], dtype=tf.float32))
            
            return (img,
                    tf.reshape(element['label'], [1]))

        return dataset.filter(lambda x: not x['is_masked_supervised'] if 'is_masked_supervised' in x else True).map(element_fn
            ).shuffle(shuffle_buffer).repeat(num_epochs).batch(batch_size)

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
            img = element['image']
            img = tf.cast(img, tf.float32)
            img = tf.math.subtract(img, tf.convert_to_tensor([255*0.49139968, 255*0.48215841, 255*0.44653091], dtype=tf.float32))
            img = tf.math.divide(img, tf.convert_to_tensor([255*0.24703223, 255*0.24348513, 255*0.26158784], dtype=tf.float32))

            return (img,
                    tf.reshape(element['label'], [1]))

        return dataset.filter(lambda x: not x['is_masked_supervised'] if 'is_masked_supervised' in x else True).map(element_fn
            ).shuffle(shuffle_buffer).repeat(num_epochs).batch(batch_size)


class DeepRotationSupervisedModel(RotationSupervisedModel):
    def __init__(self, ph):
        super(DeepRotationSupervisedModel, self).__init__(ph)

    def __call__(self):
        '''
        Returns a compiled keras model.
        '''
        model = tf.keras.models.Sequential([
                create_deep_feature_extractor_block(self.input_shape),
                create_deep_label_classifier_block(self.input_shape, self.output_shape)
            ])

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=self.optimizer(learning_rate=self.learning_rate, 
                                        nesterov=self.nesterov,
                                        momentum=self.momentum, 
                                        decay=self.decay),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
            )

        if self.pretrained_model_fp:
            model.load_weights(self.pretrained_model_fp, by_name=True)

        return model


class SimpleRotationSupervisedModel(RotationSupervisedModel):
    def __init__(self, ph):
        super(SimpleRotationSupervisedModel, self).__init__(ph)

    def __call__(self):
        '''
        Returns a compiled keras model.
        '''
        model = tf.keras.models.Sequential([
                create_simple_feature_extractor_block(self.input_shape),
                create_simple_label_classifier_block(self.input_shape, self.output_shape)
            ])

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=self.optimizer(learning_rate=self.learning_rate, 
                                        nesterov=self.nesterov,
                                        momentum=self.momentum, 
                                        decay=self.decay),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
            )

        if self.pretrained_model_fp:
            model.load_weights(self.pretrained_model_fp, by_name=True)

        return model


class RotationSelfSupervisedModel(Model):
    '''
    Predicts rotation of images
    '''
    def __init__(self, ph):
        Model.__init__(self, ph)
        self.input_shape = INPUT_SHAPES[ph['dataset']]

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
            img = tf.expand_dims(element['pixels'], 2)

            rotated_elements = (
                tf.data.Dataset.from_tensor_slices([tf.image.rot90(img, rot) for rot in [0, 1, 2, 3]]),
                tf.data.Dataset.from_tensor_slices([[0],[1],[2],[3]])
            )
            return tf.data.Dataset.zip(rotated_elements)

        return dataset.filter(lambda x: not x['is_masked_unsupervised'] if 'is_masked_unsupervised' in x else True).shuffle(
            shuffle_buffer).flat_map(element_fn).repeat(num_epochs).batch(batch_size)

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
            img = tf.cast(element['image'], tf.float32)
            img = tf.math.subtract(img, tf.convert_to_tensor([255*0.49139968, 255*0.48215841, 255*0.44653091], dtype=tf.float32))
            img = tf.math.divide(img, tf.convert_to_tensor([255*0.24703223, 255*0.24348513, 255*0.26158784], dtype=tf.float32))

            rotated_elements = (
                tf.data.Dataset.from_tensor_slices([tf.image.rot90(img, rot) for rot in [0, 1, 2, 3]]),
                tf.data.Dataset.from_tensor_slices([[0],[1],[2],[3]])
            )
            return tf.data.Dataset.zip(rotated_elements)

        return dataset.filter(lambda x: not x['is_masked_unsupervised'] if 'is_masked_unsupervised' in x else True).shuffle(
            shuffle_buffer).flat_map(element_fn).batch(batch_size).repeat(num_epochs)

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
            img = tf.cast(element['image'], tf.float32)
            img = tf.math.subtract(img, tf.convert_to_tensor([255*0.49139968, 255*0.48215841, 255*0.44653091], dtype=tf.float32))
            img = tf.math.divide(img, tf.convert_to_tensor([255*0.24703223, 255*0.24348513, 255*0.26158784], dtype=tf.float32))

            rotated_elements = (
                tf.data.Dataset.from_tensor_slices([tf.image.rot90(img, rot) for rot in [0, 1, 2, 3]]),
                tf.data.Dataset.from_tensor_slices([[0],[1],[2],[3]])
            )
            return tf.data.Dataset.zip(rotated_elements)

        return dataset.filter(lambda x: not x['is_masked_unsupervised'] if 'is_masked_unsupervised' in x else True).shuffle(
            shuffle_buffer).flat_map(element_fn).repeat(num_epochs).batch(batch_size)


class SimpleRotationSelfSupervisedModel(RotationSelfSupervisedModel):
    '''
    Predicts rotation of images
    '''
    def __init__(self, ph):
        super(SimpleRotationSelfSupervisedModel, self).__init__(ph)

    def __call__(self):
        '''
        Returns a compiled keras model.
        '''
        model = tf.keras.models.Sequential([
            create_simple_feature_extractor_block(self.input_shape),
            create_simple_rotation_classifier_block(self.input_shape, num_classes=4)
        ])

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=self.optimizer(learning_rate=self.learning_rate,
                                        nesterov=self.nesterov,
                                        momentum=self.momentum, 
                                        decay=self.decay),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model    


class DeepRotationSelfSupervisedModel(RotationSelfSupervisedModel):
    '''
    Predicts rotation of images
    '''
    def __init__(self, ph):
        super(DeepRotationSelfSupervisedModel, self).__init__(ph)

    def __call__(self):
        '''
        Returns a compiled keras model.
        '''
        model = tf.keras.models.Sequential([
            create_deep_feature_extractor_block(self.input_shape),
            create_deep_rotation_classifier_block(self.input_shape, num_classes=4)
        ])

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=self.optimizer(learning_rate=self.learning_rate,
                                        nesterov=self.nesterov,
                                        momentum=self.momentum, 
                                        decay=self.decay),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model