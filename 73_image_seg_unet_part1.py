import tensorflow as tf

IMG_WIDTH=128
IMG_HEIGHT=128
Channels= 3


inputs= tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, Channels))
s= tf.keras.layers.Lambda(lambda x: x/255)(inputs)
c1=tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs) #kernel initializer is used for starting(initialize) weights. 
c1= tf.keras.layers.Dropout(0.1)(c1)
c1=tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1= tf.keras.layers.MaxPooling2D((2, 2))(c1)


c2= tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2= tf.keras.layers.Dropout(0.1)(c2)
c2= tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2= tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3= tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3= tf.keras.layers.Dropout(0.2)(c3)
c3= tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3= tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c3)


c4= tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4= tf.keras.layers.Dropout(0.2)(c4)
c4= tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4= tf.keras.layers.MaxPooling2D((2, 2))(c4)

c5= tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_intializer='he_normal', padding='same')(p4)
c5= tf.keras.layers.Dropout(0.2)(c5)
c5= tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)


#expansion
u6= tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6= tf.keras.layers.contatenate([u6, c4])
u6= tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
u6= tf.keras.layers.Dropout(0.2)(u6)
u6= tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)

u7= tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(u6)
u7= tf.keras.layers.concatenate([u7, c3])
u7= tf.keras.layers.Conv2D(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(u7)
u7= tf.keras.layers.Dropout(0.2)(u7)
u7= tf.keras.layers.Conv2D(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(u7)

u8= tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(u7)
u8= tf.keras.layers.concatenate([u8, c2])
u8= tf.keras.layers.Conv2D(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(u8)
u8= tf.keras.layers.Dropout(0.2)(u8)
u8= tf.keras.layers.Conv2D(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(u8)

u9= tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(u8)
u9= tf.keras.layers.concatenate([u9+c1])
u9= tf.keras.layers.Conv2D(16, (2, 2), strides=(2, 2), padding='same', kernel_initialzer='he_normal')