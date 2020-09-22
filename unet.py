from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf
# from keras import backend as K
# from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout
# from keras.models import Model
# from keras.optimizers import Adam

from constants import img_rows, img_cols

# tf.enable_eager_execution()
tf.keras.backend.set_image_data_format('channels_last')

smooth = 1

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
    
@tf.function
def call(self, inputs, training=None):
    if 0. < self.rate < 1.:
        noise_shape = self._get_noise_shape(inputs)
        def dropped_inputs():
            return tf.keras.backend.dropout(inputs, self.rate, noise_shape, seed=self.seed)
        if(training):
            return tf.keras.backend.in_train_phase(dropped_inputs, inputs, training=training)
        else:
            return tf.keras.backend.in_test_phase(dropped_inputs, inputs, training=None)
    return inputs

tf.keras.layers.Dropout.call = call

# 
def get_unet(dropout):
    inputs = tf.keras.layers.Input((img_rows, img_cols, 1))
    conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same',input_shape=inputs.shape)(inputs)
    conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)


    conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

   
    conv3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
   
    conv5 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv5)

    if dropout:
        conv5 = tf.keras.layers.Dropout(0.5)(conv5)    

    up6 = tf.keras.layers.Concatenate(axis=-1)([tf.keras.layers.UpSampling2D(size=(2, 2))(conv5), conv4])

    conv6 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv6)

    up7 = tf.keras.layers.Concatenate(axis=-1)([tf.keras.layers.UpSampling2D(size=(2, 2))(conv6), conv3])

    conv7 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv7)

    up8 = tf.keras.layers.Concatenate(axis=-1)([tf.keras.layers.UpSampling2D(size=(2, 2))(conv7), conv2])

    conv8 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up8)
    conv8 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv8)

    up9 = tf.keras.layers.Concatenate(axis=-1)([tf.keras.layers.UpSampling2D(size=(2, 2))(conv8), conv1])

    conv9 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(up9)
    conv9 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv9)

    conv10 = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    my_loss= dice_coef_loss
    my_metric = dice_coef

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5), loss=my_loss, metrics=[my_metric])

    return model

if __name__ == "__main__":
    get_unet(True)