# import random

# def predict_random_result(image_data):
#    return random.randint(0, 1)

###########################################################

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
# from keras.layers import BatchNormalization

ds_path = '/content/drive/MyDrive/alpaca_dataset '

train_generator = ImageDataGenerator(rescale=1./255.)
test_generator = ImageDataGenerator(rescale=1./255.,validation_split=.2)

train_ds = train_generator.flow_from_directory(ds_path, 
                                               classes={'not alpaca': 0, 
                                                        'alpaca': 1},
                                               batch_size = 64,
                                               class_mode='binary', 
                                               target_size = (256,256), 
                                               subset='training')

valid_ds = test_generator.flow_from_directory(ds_path, 
                                              classes={'not alpaca': 0, 
                                                        'alpaca': 1},
                                               batch_size = 64, 
                                               class_mode='binary', 
                                               target_size = (256,256), 
                                               subset='validation')

model = keras.Sequential()

# Convolutional layer and maxpool layer 1
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding='same', input_shape = ((256,256,3))))
model.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2)))

# Convolutional layer and maxpool layer 2
model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding='same'))
model.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2)))

# Convolutional layer and maxpool layer 3
model.add(Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding='same'))
model.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2)))

# model.add(SeqSelfAttention(attention_activation='sigmoid'))

# Flattening Operation
model.add(Flatten())

# Fully Connected layer
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dropout(rate = 0.5))

# Output layer
model.add(Dense(units = 1,activation='sigmoid'))

model.compile(optimizer='adam', # optimizer=optimizers.RMSprop(lr=0.001)
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])


# Early Stopping 
Early_Stopping = tf.keras.callbacks.EarlyStopping(
                                                  monitor="val_loss",
                                                  patience=5,
                                                  min_delta=1e-4,
                                                  restore_best_weights=True)

# Training the model
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=50,
    callbacks=[Early_Stopping])

# Prediction
def Alpaca_Prediction(image_data):
    test_image = image.load_img(image_data, target_size = (256, 256))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    return (result >= 0.5).astype(np.int32)

