import tensorflow as tf
import numpy as np
import os
from keras.preprocessing import image


class LetterReader:

    def __init__(self, training_data=None, test_data=None):
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
        self.cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3,
                                            activation='relu'))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))

        # Step 3 - Flattening
        self.cnn.add(tf.keras.layers.Flatten())

        # Step 4 - Full Connection
        self.cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))
        self.cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

        # Lets add a dropout layer too
        self.cnn.add(tf.keras.layers.Dropout(0.2))

        # Step 5 - Output Layer
        # no. of output nodes = no of datasets
        self.cnn.add(tf.keras.layers.Dense(
            units=len(os.listdir("dataset/letters/training_set/")),
            activation="softmax"))

    def train(self):
        self.cnn.compile(optimizer="adam", loss='categorical_crossentropy',
                         metrics=['accuracy'])
        self.cnn.fit(x=self.training_set, validation_data=self.test_set,
                     epochs=5)

    def save(self):
        self.cnn.save('model.h5')

    def load(self, model):
        self.cnn = tf.keras.models.load_model(model)

    def prediction(self, pic):
        """
        Takes the image and transforms it and return the one - hot encoding result
        :param pic:
        :return:
        """
        test_image = image.load_img(pic, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        return self.cnn.predict(test_image)


# L = LetterReader()
# L.load('model.h5')
# files = os.listdir("dataset/letters/test_set/")
# index_map = dict()
# for n in range(len(os.listdir("dataset/letters/test_set/"))):
#     index_map[n] = files[n]
#
# encoding = L.prediction("dataset/letters/single_prediction/ROI_3.PNG")
#
# print(encoding)
# for i in range(len(encoding[0])):
#     if encoding[0][i] >= 0.9:
#         print(index_map[i])
