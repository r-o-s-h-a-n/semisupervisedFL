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



NCHANNELS1 = 192
NCHANNELS2 = 160
NCHANNELS3 = 96

INPUT_SHAPES = {'emnist': (28,28,1), 'cifar100': (32,32,3), 'cifar10central': (32,32,3)}
OUTPUT_SHAPES = {'emnist': 10, 'cifar100': 100, 'cifar10central': 10}


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


class SoftmaxCrossEntropyLoss(tf.keras.losses.Loss):
    def __call__(self, y_true, y_pred):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(y_true,y_pred)


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


def create_NIN_block(out_planes, kernel_size, name=None):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            out_planes, 
            kernel_size=kernel_size, 
            strides=1, 
            padding='same', 
            use_bias=False,
            kernel_initializer=ConvInitializer(kernel_size, out_planes)
            ),
        tf.keras.layers.BatchNormalization(
            axis=-1,
            epsilon=1E-5,
            momentum=0.1
            ),
        tf.keras.layers.ReLU()
    ], name=name)
    return  model


class GlobalAveragePooling(tf.keras.layers.Layer):
    def __init__(self, name):
        super(GlobalAveragePooling, self).__init__(name=name)
    
    def build(self, input_shape):
        self.avgpool = tf.keras.layers.AveragePooling2D(
                                        pool_size=(input_shape[1], input_shape[2]), padding='valid')
        self.reshape = tf.keras.layers.Reshape((-1, input_shape[-1]))

    def call(self, input):
        x = self.avgpool(input)
        x = self.reshape(x)
        return x


def create_feature_extractor_block(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape),
        # block 1
        create_NIN_block(NCHANNELS1, 5, 'Block1_Conv1'),
        create_NIN_block(NCHANNELS2, 1, 'Block1_Conv2'),
        create_NIN_block(NCHANNELS3, 1, 'Block1_Conv3'),
        tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='same', name='Block1_MaxPool'),

        # block 2
        create_NIN_block(NCHANNELS1, 5, 'Block2_Conv1'),
        create_NIN_block(NCHANNELS1, 1, 'Block2_Conv2'),
        create_NIN_block(NCHANNELS1, 1, 'Block2_Conv3'),
        tf.keras.layers.AveragePooling2D(pool_size=3,strides=2,padding='same', name='Block2_AvgPool')
    ], name='Conv_Feature_Extractor')

    return model


def create_conv_label_classifier_block(num_classes=10):
    model = tf.keras.models.Sequential([
        # block 3
        create_NIN_block(NCHANNELS1, 3, 'Block3_Conv1'),
        create_NIN_block(NCHANNELS1, 1, 'Block3_Conv2'),
        create_NIN_block(NCHANNELS1, 1, 'Block3_Conv3'),
        # tf.keras.layers.AveragePooling2d(pool_size=3,strides=2,padding=1, name='Block3_AvgPool'), 

        # # block 4
        # create_NIN_block(nChannels1, 3, 'Block4_Conv1'),
        # create_NIN_block(nChannels1, 1, 'Block4_Conv2'),
        # create_NIN_block(nChannels1, 1, 'Block4_Conv3'),

        # # block 5
        # create_NIN_block(nChannels1, 3, 'Block5_Conv1'),
        # create_NIN_block(nChannels1, 1, 'Block5_Conv2'),
        # create_NIN_block(nChannels1, 1, 'Block5_Conv3'),

        GlobalAveragePooling(name='Global_Avg_Pool'),
        tf.keras.layers.Dense(num_classes, name='Linear_Classifier')
    ],
    name = 'Label_Classifier')
    return model


def create_conv_rotation_classifier_block(num_classes=4):
    model = tf.keras.models.Sequential([
        # block 3
        create_NIN_block(NCHANNELS1, 3, 'Block3_Conv1'),
        create_NIN_block(NCHANNELS1, 1, 'Block3_Conv2'),
        create_NIN_block(NCHANNELS1, 1, 'Block3_Conv3'),
        # tf.keras.layers.AveragePooling2D(pool_size=3,strides=2,padding='same', name='Block3_AvgPool'), 

        # block 4
        create_NIN_block(NCHANNELS1, 3, 'Block4_Conv1'),
        create_NIN_block(NCHANNELS1, 1, 'Block4_Conv2'),
        create_NIN_block(NCHANNELS1, 1, 'Block4_Conv3'),

        # # block 5
        # create_NIN_block(NCHANNELS1, 3, 'Block5_Conv1'),
        # create_NIN_block(NCHANNELS1, 1, 'Block5_Conv2'),
        # create_NIN_block(NCHANNELS1, 1, 'Block5_Conv3'),

        GlobalAveragePooling(name='Global_Avg_Pool'),
        # tf.keras.layers.Dense(num_classes, name='Linear_Classifier', activation='softmax')
        tf.keras.layers.Dense(num_classes, name='Linear_Classifier')

    ],
    name = 'Rot_Classifier')
    return model


def create_linear_label_classifier_block():
    raise NotImplementedError


def create_linear_rotation_classifier_block():
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


def rotate_img_tensor(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return tf.image.rot90(img)
    elif rot == 180: # 180 degrees rotation
        return tf.image.flip_left_right(tf.image.flip_up_down(img))
    elif rot == 270: # 270 degrees rotation / or -90
        return tf.image.transpose(tf.image.flip_up_down(img))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class RotationSupervisedModel(Model):
    def __init__(self, ph):
        Model.__init__(self, ph)
        self.output_shape = OUTPUT_SHAPES[self.ph['dataset']]
        self.pretrained_model_fp = self.ph.setdefault('pretrained_model_fp', None)

    def __call__(self):
        '''
        Returns a compiled keras model.
        '''
        feature_extractor = create_feature_extractor_block()

        if self.pretrained_model_fp:
            feature_extractor.load_weights(self.pretrained_model_fp, by_name=True)

        model = tf.keras.models.Sequential([
                feature_extractor,
                create_conv_label_classifier_block(self.output_shape)
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
            img = tf.math.divide(tf.cast(element['image'], tf.float32),
                                tf.constant(255.0, dtype=tf.float32))

            return (img,
                    tf.reshape(element['label'], [1]))

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
            img = tf.math.divide(tf.cast(element['image'], tf.float32),
                                tf.constant(255.0, dtype=tf.float32))

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
        self.input_shape = INPUT_SHAPES[ph['dataset']]

    def __call__(self):
        '''
        Returns a compiled keras model.
        '''
        model = tf.keras.models.Sequential([
            create_feature_extractor_block(self.input_shape),
            create_conv_rotation_classifier_block()
        ])

        model.compile(
            # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            loss = SoftmaxCrossEntropyLoss(),
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
                tf.data.Dataset.from_tensor_slices([0,1,2,3])
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
            img = tf.math.subtract(img, tf.convert_to_tensor([0.49139968, 0.48215841, 0.44653091], dtype=tf.float32))
            img = tf.math.divide(img, tf.convert_to_tensor([0.24703223, 0.24348513, 0.26158784], dtype=tf.float32))

            rotated_elements = (
                tf.data.Dataset.from_tensor_slices([rotate_img_tensor(img, rot) for rot in [0, 90, 180, 270]]),
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
            # img = tf.math.divide(tf.cast(element['image'], tf.float32),
                                # tf.constant(255.0, dtype=tf.float32))
            img = tf.cast(element['image'], tf.float32)
            img = tf.math.subtract(img, tf.convert_to_tensor([255*0.49139968, 255*0.48215841, 255*0.44653091], dtype=tf.float32))
            img = tf.math.divide(img, tf.convert_to_tensor([255*0.24703223, 255*0.24348513, 255*0.26158784], dtype=tf.float32))

            rotated_elements = (
                tf.data.Dataset.from_tensor_slices([rotate_img_tensor(img, rot) for rot in [0, 90, 180, 270]]),
                # tf.data.Dataset.from_tensor_slices([0,1,2,3])
                # tf.data.Dataset.from_tensor_slices([0.0,1.0,2.0,3.0])
                tf.data.Dataset.from_tensor_slices([[0],[1],[2],[3]])
            )
            return tf.data.Dataset.zip(rotated_elements)

        return dataset.filter(lambda x: not x['is_masked_unsupervised'] if 'is_masked_unsupervised' in x else True).shuffle(
            shuffle_buffer).flat_map(element_fn).repeat(num_epochs).batch(batch_size)