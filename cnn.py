import tensorflow as tf
import numpy as np
from image_preprocessor import ImageProcessor
from tensorflow.keras.optimizers import RMSprop


class LetterReader:

    def __init__(self, training_data, test_data):
        self.cnn = None
        self.training_set = training_data
        self.test_set = test_data

    def create_model(self):
        self.cnn = tf.keras.models.Sequential()

        # Step 1 - Convolution
        self.cnn.add(
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                                   input_shape=[64, 64, 3]))

        # Step 2 - Pooling
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))

        # Adding a second convolutional layer
        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,
                                            activation='relu'))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))

        # Adding a third convolutional layer
        # self.cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3,
        #                                     activation='relu'))
        # self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))


        # Step 3 - Flattening
        self.cnn.add(tf.keras.layers.Flatten())

        # Step 4 - Full Connection
        self.cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


        # Step 5 - Output Layer
        # currently we have 13 outputs -> 13 nodes
        self.cnn.add(tf.keras.layers.Dense(units=13, activation="softmax"))

    def train(self):
        self.cnn.compile(optimizer=RMSprop(lr=0.01), loss='categorical_crossentropy',
                         metrics=['accuracy'])
        self.cnn.fit(x=self.training_set, validation_data=self.test_set,
                     epochs=2)

    def save(self):
        self.cnn.save('model.h5')



