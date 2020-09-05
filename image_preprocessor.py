import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


class ImageProcessor:
    def __init__(self):
        self.training_data = None
        self.test_data = None

    def process_training_data(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.3,
                                           horizontal_flip=True)
        training_set = train_datagen.flow_from_directory(
            './dataset/letters/training_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')

        self.training_data = training_set

    def process_test_data(self):
        # Preprocessing the Test set
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_set = test_datagen.flow_from_directory(
            './dataset//letters/test_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')
        self.test_data = test_set

    def get_training_set(self):
        if self.training_data is not None:
            return self.training_data
        else:
            self.process_training_data()
            return self.training_data

    def get_testing_set(self):
        if self.test_data is not None:
            return self.training_data
        else:
            self.process_test_data()
            return self.test_data

