
import tensorflow as tf
import tensorflow_addons as tfa
from models.initializers import ConvInitializer


def create_NIN_block(out_planes, kernel_size, strides=1, padding='same', name=None, input_shape=None):
    if input_shape:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                out_planes, 
                kernel_size=kernel_size, 
                strides=strides, 
                padding=padding, 
                input_shape=input_shape,
                use_bias=False,
                kernel_initializer=ConvInitializer(kernel_size, out_planes),
                ),
            tfa.layers.InstanceNormalization(axis=-1),
            tf.keras.layers.ReLU()
        ], name=name)
        return  model

    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                out_planes, 
                kernel_size=kernel_size, 
                strides=1, 
                padding='same', 
                use_bias=False,
                kernel_initializer=ConvInitializer(kernel_size, out_planes),
                ),
            tfa.layers.InstanceNormalization(axis=-1),
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
