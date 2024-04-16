from tensorflow.keras.layers import *
from tensorflow.keras.models import Model



def conv_block(num_filters, inputs):
    x= Conv2D(num_filters, 3, padding='same')(inputs)
    x= BatchNormalization()(x)
    x= Activation("relu")(x)

    x= Conv2D(num_filters, 3, padding='same')(x)
    x= BatchNormalization()(x)
    x= Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    x= conv_block(num_filters, inputs)
    p=MaxPooling2D((2, 2))(x)
    return x, p


def decoder_block(inputs, p, num_filters):
    u= Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    print(u.shape,'======', p.shape)
    u= concatenate([u, p])
    c= conv_block(num_filters, u)
    return c

def build_unet(input_shape):
    inputs= Input(input_shape)
    c1, p1= encoder_block(inputs, 16)
    c2, p2= encoder_block(p1, 32)
    c3, p3= encoder_block(p2, 64)
    c4, p4= encoder_block(p3, 128)
    p5= conv_block(256, p4)
    c6=decoder_block(p5, c4, 128)
    c7=decoder_block(c6, c3, 64)
    c8=decoder_block(c7, c2, 32)
    c9=decoder_block(c8, c1, 16)

    outputs= Conv2D(1, 1, padding='same', activation='sigmoid')(c9)
    print(outputs)

    model= Model(inputs, outputs, name='UNET')
    return model

model= build_unet((128, 128, 3))
print(model.summary())
