import numpy as np
import tensorflow as tf


def rotate_img_numpy(img, rot):
    '''
    Rotates an image in numpy.
    '''
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
    '''
    Rotates an image in tensorflow.
    '''
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