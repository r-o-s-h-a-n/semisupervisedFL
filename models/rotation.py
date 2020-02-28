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

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
import tensorflow_addons as tfa


# NCHANNELS1 = 192
# NCHANNELS2 = 160
# NCHANNELS3 = 96


NCHANNELS1 = 32
NCHANNELS2 = 64


CHANNELS_LAST = True

INPUT_SHAPES_CHANNELS_FIRST = {'emnist': [1,28,28], 'cifar100': [3,32,32], 'cifar10central': [3,32,32]}
INPUT_SHAPES_CHANNELS_LAST = {'emnist': [28,28,1], 'cifar100': [32,32,3], 'cifar10central': [32,32,3]}
OUTPUT_SHAPES = {'emnist': 10, 'cifar100': 20, 'cifar10central': 10}


def _assert_float_dtype(dtype):
  """Validate and return floating point type based on `dtype`.
  `dtype` must be a floating point type.
  Args:
    dtype: The data type to validate.
  Returns:
    Validated type.
  Raises:
    ValueError: if `dtype` is not a floating point type.
  """
  if not dtype.is_floating:
    raise ValueError("Expected floating point type, got %s." % dtype)
  return dtype


class ConvInitializer(tf.keras.initializers.Initializer):
    '''
    Conv weight initializer used by Rotation Net paper.
    '''
    def __init__(self, 
                kernel_size,
                out_channels,
                seed=None, 
                dtype=dtypes.float32):
        assert (isinstance(kernel_size, int) or (isinstance(kernel_size, tuple) and len(kernel_size)==2)), 'kernel size must be int or tuple'

        super(ConvInitializer, self).__init__()
        self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.seed = seed

    def __call__(self, shape, dtype=None):
        if dtype is None:
            dtype = self.dtype
        
        if isinstance(self.kernel_size, int):
            n = self.kernel_size * self.kernel_size * self.out_channels
        else:
            n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        stddev = math.sqrt(2/n)
        return random_ops.random_normal(shape, 0.0, stddev, dtype, seed=self.seed)

    def get_config(self):
        return {
            "kernel_size": self.kernel_size,
            "out_channels": self.out_channels,
            "seed": self.seed,
            "dtype": self.dtype.name
        }

class DenseInitializer(tf.keras.initializers.Initializer):
    '''
    Dense weight initializer used by Rotation Net paper.
    '''
    def __init__(self, 
                out_channels,
                seed=None, 
                dtype=dtypes.float32):

        super(DenseInitializer, self).__init__()
        self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))
        self.out_channels = out_channels
        self.seed = seed

    def __call__(self, shape, dtype=None):
        if dtype is None:
            dtype = self.dtype
        
        stddev = math.sqrt(2.0 / self.out_channels)
        return random_ops.random_normal(shape, 0.0, stddev, dtype, seed=self.seed)

    def get_config(self):
        return {
            "out_channels": self.out_channels,
            "seed": self.seed,
            "dtype": self.dtype.name
        }

# def create_NIN_block(out_planes, kernel_size, name=None, trainable=True, channels_last=True):
#     if channels_last:
#         data_format = 'channels_last'
#         axis = -1
#     else:
#         data_format = 'channels_first'
#         axis = 1

#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(
#             out_planes, 
#             kernel_size=kernel_size, 
#             strides=1, 
#             padding='same', 
#             use_bias=False,
#             kernel_initializer=ConvInitializer(kernel_size, out_planes),
#             trainable=trainable,
#             data_format=data_format
#             ),
#         tfa.layers.InstanceNormalization(axis=-1),
#         tf.keras.layers.ReLU()
#     ], name=name)
#     return  model


# class GlobalAveragePooling(tf.keras.layers.Layer):
#     def __init__(self, name, channels_last=True):
#         super(GlobalAveragePooling, self).__init__(name=name)
#         self.channels_last = channels_last
    
#     def build(self, input_shape):
#         if self.channels_last:
#             self.avgpool = tf.keras.layers.AveragePooling2D(
#                                             pool_size=(input_shape[1], input_shape[2]), padding='valid')
#             self.reshape = tf.keras.layers.Reshape((-1, input_shape[-1]))
#         else:
#             self.avgpool = tf.keras.layers.AveragePooling2D(
#                                             pool_size=(input_shape[2], input_shape[3]), padding='valid')
#             self.reshape = tf.keras.layers.Reshape((-1, input_shape[1]))

#     def call(self, input):
#         x = self.avgpool(input)
#         x = self.reshape(x)
#         return x


# def create_feature_extractor_block(input_shape, trainable=True, channels_last=True):
#     if channels_last:
#         data_format = 'channels_last'
#     else:
#         data_format = 'channels_first'

#     model = tf.keras.models.Sequential([
#         tf.keras.layers.InputLayer(input_shape),
#         # block 1
#         create_NIN_block(NCHANNELS1, 5, 'Block1_Conv1', trainable=trainable, channels_last=channels_last),
#         create_NIN_block(NCHANNELS2, 1, 'Block1_Conv2', trainable=trainable, channels_last=channels_last),
#         create_NIN_block(NCHANNELS3, 1, 'Block1_Conv3', trainable=trainable, channels_last=channels_last),
#         tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='same', name='Block1_MaxPool', data_format=data_format),

#         # block 2
#         create_NIN_block(NCHANNELS1, 5, 'Block2_Conv1', trainable=trainable, channels_last=channels_last),
#         create_NIN_block(NCHANNELS1, 1, 'Block2_Conv2', trainable=trainable, channels_last=channels_last),
#         create_NIN_block(NCHANNELS1, 1, 'Block2_Conv3', trainable=trainable, channels_last=channels_last),
#         tf.keras.layers.AveragePooling2D(pool_size=3,strides=2,padding='same', name='Block2_AvgPool', data_format=data_format)
#     ], name='Conv_Feature_Extractor')

#     return model


# def create_conv_label_classifier_block(num_classes=10, channels_last=True):
#     model = tf.keras.models.Sequential([
#         # block 3
#         create_NIN_block(NCHANNELS1, 3, 'Block3_Conv1', channels_last=channels_last),
#         create_NIN_block(NCHANNELS1, 1, 'Block3_Conv2', channels_last=channels_last),
#         create_NIN_block(NCHANNELS1, 1, 'Block3_Conv3', channels_last=channels_last),

#         GlobalAveragePooling(name='Global_Avg_Pool', channels_last=channels_last),
#         tf.keras.layers.Dense(num_classes, 
#                                 name='Linear_Classifier', 
#                                 activation='softmax',
#                                 kernel_initializer=DenseInitializer(num_classes))
#     ],
#     name = 'Label_Classifier')
#     return model


# def create_conv_rotation_classifier_block(num_classes=4, channels_last=True):
#     model = tf.keras.models.Sequential([
#         # block 3
#         create_NIN_block(NCHANNELS1, 3, 'Block3_Conv1', channels_last=channels_last),
#         create_NIN_block(NCHANNELS1, 1, 'Block3_Conv2', channels_last=channels_last),
#         create_NIN_block(NCHANNELS1, 1, 'Block3_Conv3', channels_last=channels_last),

#         # block 4
#         create_NIN_block(NCHANNELS1, 3, 'Block4_Conv1', channels_last=channels_last),
#         create_NIN_block(NCHANNELS1, 1, 'Block4_Conv2', channels_last=channels_last),
#         create_NIN_block(NCHANNELS1, 1, 'Block4_Conv3', channels_last=channels_last),

#         GlobalAveragePooling(name='Global_Avg_Pool', channels_last=channels_last),
#         tf.keras.layers.Dense(num_classes,
#                                 name='Linear_Classifier', 
#                                 activation='softmax',
#                                 kernel_initializer=DenseInitializer(num_classes))
#     ],
#     name = 'Rot_Classifier')
#     return model


def create_simple_feature_extractor_block(input_shape, trainable=True, channels_last=True):
    model = tf.keras.models.Sequential([
        # block 1
        tf.keras.layers.Conv2D(NCHANNELS1, 5, padding='same', name='sam', input_shape=input_shape),
        tfa.layers.InstanceNormalization(axis=-1),
        tf.keras.layers.ReLU()
    ], name='Conv_Feature_Extractor')

    return model


def create_simple_label_classifier_block(input_shape, num_classes=10, channels_last=True):
    input_shape = input_shape[:2] + [NCHANNELS1]
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(NCHANNELS2, 5, name='frodo', input_shape=input_shape),
        tfa.layers.InstanceNormalization(axis=-1),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, 
                                name='Hidden_Layer', 
                                activation='relu',
                                ),
        tf.keras.layers.Dense(num_classes, 
                                name='Linear_Classifier', 
                                activation='softmax',
                                )
    ],
    name = 'Label_Classifier')
    return model


def create_simple_rotation_classifier_block(input_shape, num_classes=4, channels_last=True):
    input_shape = input_shape[:2] + [NCHANNELS1]

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(NCHANNELS2, 5, input_shape=input_shape),
        tfa.layers.InstanceNormalization(axis=-1),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, 
                                name='Hidden_Layer', 
                                activation='relu',
                                ),
        tf.keras.layers.Dense(num_classes, 
                                name='Linear_Classifier', 
                                activation='softmax',
                                )
    ],
    name = 'Label_Classifier')
    return model


def create_dense_label_classifier_block():
    raise NotImplementedError


def create_dense_rotation_classifier_block():
    raise NotImplementedError


def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2)))
    elif rot == 180: # 180 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


def rotate_img_tensor(img, rot, channels_last=True):
    if not channels_last:
        img = tf.transpose(img, [1, 2, 0]) # convert NCHW to NHWC

    if rot == 0: # 0 degrees rotation
        res = img
    elif rot == 90: # 90 degrees rotation
        res = tf.image.rot90(img)
    elif rot == 180: # 180 degrees rotation
        res = tf.image.flip_left_right(tf.image.flip_up_down(img))
    elif rot == 270: # 270 degrees rotation / or -90
        res = tf.image.transpose(tf.image.flip_up_down(img))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

    if not channels_last:
        return tf.transpose(res, [2, 0, 1])
    else:
        return res


class RotationSupervisedModel(Model):
    def __init__(self, ph):
        Model.__init__(self, ph)
        if CHANNELS_LAST:
            self.input_shape = INPUT_SHAPES_CHANNELS_LAST[ph['dataset']]
        else:
            self.input_shape = INPUT_SHAPES_CHANNELS_FIRST[ph['dataset']]
        self.channels_last = CHANNELS_LAST
        self.output_shape = OUTPUT_SHAPES[self.ph['dataset']]
        self.pretrained_model_fp = self.ph.setdefault('pretrained_model_fp', None)
        self.fine_tune_feature_extractor = self.ph.setdefault('fine_tune_feature_extractor', True)

        if self.pretrained_model_fp:
            tf.print('training on a pretrained model')

    def __call__(self):
        '''
        Returns a compiled keras model.
        '''
        feature_extractor = create_simple_feature_extractor_block(self.input_shape, 
                                                        trainable=self.fine_tune_feature_extractor,
                                                        channels_last=self.channels_last)

        if self.pretrained_model_fp:
            feature_extractor.load_weights(self.pretrained_model_fp, by_name=True)

        model = tf.keras.models.Sequential([
                feature_extractor,
                create_simple_label_classifier_block(self.input_shape, self.output_shape, channels_last=self.channels_last)
            ])
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=self.optimizer(learning_rate=self.learning_rate, 
                                        nesterov=self.nesterov,
                                        momentum=self.momentum, 
                                        decay=self.decay),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
            )
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
            return (tf.expand_dims(element['pixels'], 2),
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
            img = element['image']
            img = tf.cast(img, tf.float32)
            # img = tf.transpose(img, [1, 2, 0]) # convert NCHW to NHWC
            img = tf.math.subtract(img, tf.convert_to_tensor([255*0.49139968, 255*0.48215841, 255*0.44653091], dtype=tf.float32))
            img = tf.math.divide(img, tf.convert_to_tensor([255*0.24703223, 255*0.24348513, 255*0.26158784], dtype=tf.float32))
            
            if not self.channels_last:
                img = tf.transpose(img, [2, 0, 1]) # convert NHWC to NCHW

            return (img,
                    tf.reshape(element['coarse_label'], [1]))

        return dataset.filter(lambda x: not x['is_masked_supervised'] if 'is_masked_supervised' in x else True).repeat(
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
            img = element['image']
            img = tf.cast(img, tf.float32)
            if not self.channels_last:
                img = tf.transpose(img, [1, 2, 0]) # convert NCHW to NHWC
            img = tf.math.subtract(img, tf.convert_to_tensor([255*0.49139968, 255*0.48215841, 255*0.44653091], dtype=tf.float32))
            img = tf.math.divide(img, tf.convert_to_tensor([255*0.24703223, 255*0.24348513, 255*0.26158784], dtype=tf.float32))
            if not self.channels_last:
                img = tf.transpose(img, [2, 0, 1]) # convert NHWC to NCHW

            return (img,
                    tf.reshape(element['label'], [1]))

        return dataset.filter(lambda x: not x['is_masked_supervised'] if 'is_masked_supervised' in x else True).repeat(
            num_epochs).map(element_fn).shuffle(shuffle_buffer).batch(batch_size)



class RotationSelfSupervisedModel(Model):
    '''
    Predicts rotation of images
    '''
    def __init__(self, ph):
        Model.__init__(self, ph)
        if CHANNELS_LAST:
            self.input_shape = INPUT_SHAPES_CHANNELS_LAST[ph['dataset']]
        else:
            self.input_shape = INPUT_SHAPES_CHANNELS_FIRST[ph['dataset']]

    def __call__(self):
        '''
        Returns a compiled keras model.
        '''
        model = tf.keras.models.Sequential([
            create_simple_feature_extractor_block(self.input_shape, trainable=True, channels_last=CHANNELS_LAST),
            create_simple_rotation_classifier_block(self.input_shape, num_classes=4, channels_last=CHANNELS_LAST)
        ])

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=self.optimizer(learning_rate=self.learning_rate,
                                        nesterov=self.nesterov,
                                        momentum=self.momentum, 
                                        decay=self.decay),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
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
            img = tf.expand_dims(element['pixels'], 2)

            rotated_elements = (
                tf.data.Dataset.from_tensor_slices([rotate_img_tensor(img, rot) for rot in [0, 90, 180, 270]]),
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
            # img = tf.transpose(img, [2, 0, 1]) # convert NHWC to NCHW

            rotated_elements = (
                tf.data.Dataset.from_tensor_slices([rotate_img_tensor(img, rot, channels_last=False) for rot in [0, 90, 180, 270]]),
                tf.data.Dataset.from_tensor_slices([[0],[1],[2],[3]])
            )
            return tf.data.Dataset.zip(rotated_elements)

        return dataset.filter(lambda x: not x['is_masked_unsupervised'] if 'is_masked_unsupervised' in x else True).shuffle(
            shuffle_buffer).flat_map(element_fn).repeat(num_epochs).batch(batch_size)

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
            # img = tf.transpose(img, [1, 2, 0]) # convert NCHW to NHWC
            img = tf.math.subtract(img, tf.convert_to_tensor([255*0.49139968, 255*0.48215841, 255*0.44653091], dtype=tf.float32))
            img = tf.math.divide(img, tf.convert_to_tensor([255*0.24703223, 255*0.24348513, 255*0.26158784], dtype=tf.float32))
            # img = tf.transpose(img, [2, 0, 1]) # convert NHWC to NCHW


            rotated_elements = (
                tf.data.Dataset.from_tensor_slices([rotate_img_tensor(img, rot, channels_last=False) for rot in [0, 90, 180, 270]]),
                tf.data.Dataset.from_tensor_slices([[0],[1],[2],[3]])
            )
            return tf.data.Dataset.zip(rotated_elements)

        return dataset.filter(lambda x: not x['is_masked_unsupervised'] if 'is_masked_unsupervised' in x else True).shuffle(
            shuffle_buffer).flat_map(element_fn).repeat(num_epochs).batch(batch_size)