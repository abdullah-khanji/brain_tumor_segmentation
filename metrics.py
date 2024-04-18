import numpy as np, tensorflow as tf
from tensorflow.keras import backend as B

smooth=1e-15

@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred):
    y_true= tf.keras.layers.Flatten()(y_true)
    y_pred= tf.keras.layers.Flatten()(y_pred)
    intersection= tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth)/(tf.reduce_sum(y_true)+tf.reduce_sum(y_pred))


@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1.0- dice_coef(y_true, y_pred)