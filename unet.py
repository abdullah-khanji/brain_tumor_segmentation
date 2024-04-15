from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def conv_block(inputs, num_filter):

    x= Conv2D(num_filter, 3, padding='same')(inputs)
    x= BatchNormalization()(x)
    x= Activation("relu")(x)

    x= Conv2D(num_filter, 3, padding='same')(x)
    x= BatchNormalization()(x)
    x= Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    x= conv_block(inputs, num_filters)
    p=MaxPool2D((2, 2))(x)
    return x, p

def build_unet(input_shape):
    inputs= Input(input_shape)
    s1, p1= encoder_block(inputs, 64)
    s2, p2= encoder_block(p1, 128)
    s3, p3= encoder_block(p2, 256)
    s4, p4= encoder_block(p3, 512)

    print(s1.shape, s2.shape, s3.shape, s4.shape)
    
build_unet((256, 256, 3))