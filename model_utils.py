# -*- coding: utf-8 -*-

import tensorflow as tf
from keras import backend as K

def sum_squared_error(y_true, y_pred):
    return (K.sum(K.square(y_pred - y_true))/2)

def ssim(y_true,y_pred):
    return tf.image.ssim(y_true,y_pred,1.0)

def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    y_pred = K.clip(y_pred, 0.0, 1.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))