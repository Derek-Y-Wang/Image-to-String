from cnn import LetterReader
from image_preprocessor import ImageProcessor
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os

files = os.listdir("dataset/letters/test_set/")
index_map = dict()
for n in range(len(os.listdir("dataset/letters/test_set/"))):
    index_map[n] = files[n]


def setup():
    data = ImageProcessor()
    data.process_training_data()
    data.process_test_data()

    cnn = LetterReader(data.get_training_set(), data.get_testing_set())
    cnn.create_model()
    cnn.train()
    cnn.save()

    # test_image = image.load_img("dataset/letters/single_prediction/letter_k.png", target_size=(64, 64))
    # test_image = image.img_to_array(test_image)
    # test_image = np.expand_dims(test_image, axis=0)
    # result = cnn.cnn.predict(test_image)
    # print(result)
    # for i in range(len(result[0])):
    #     if result[0][i] == 1:
    #         print(index_map[i])


def load_results(model, image_path):

    cnn = tf.keras.models.load_model(model)
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)
    # training_set.class_indices
    print(result)
    for i in range(len(result[0])):
        if result[0][i] == 1:
            print(index_map[i])


if __name__ == "__main__":

    while True:
        command = input("Type 'train' to train model or 'load' to get results: \n")
        if command == "train":
            setup()
        elif command == "load":
            IMAGE_PATH = "dataset/letters/single_prediction/"
            pic = input("Name of image: \n")
            IMAGE_PATH += pic
            load_results('model.h5', IMAGE_PATH)
        elif command == "quit":
            break

