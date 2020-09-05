from cnn import LetterReader
from image_preprocessor import ImageProcessor
import tensorflow as tf
from keras.preprocessing import image
import numpy as np


def setup():
    data = ImageProcessor()
    data.process_training_data()
    data.process_test_data()

    cnn = LetterReader(data.get_training_set(), data.get_testing_set())
    cnn.create_model()
    cnn.train()
    # cnn.save()

    test_image = image.load_img("dataset/letters/single_prediction/letter_k.png", target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.cnn.predict(test_image)
    print(result)
    if result[0][0] == 1:
        prediction = 'J'
    elif result[0][1] == 1:
        prediction = 'K'
    elif result[0][2] == 1:
        prediction = 'L'
    else:
        prediction = 'M'
    print(prediction)


def load_results(model, image_path):

    cnn = tf.keras.models.load_model(model)
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)
    # training_set.class_indices
    if result[0][0] == 1:
        prediction = 'J'
    elif result[0][0] == 1:
        prediction = 'K'
    elif result[0][0] == 1:
        prediction = 'L'
    else:
        prediction = 'M'
    print(prediction)


if __name__ == "__main__":
    setup()
    # load_results('model.h5', "dataset/letters/single_prediction/paint_M.png")

    # print(cnn.classify("dataset/letters/single_prediction/train_4b_00105.png"))
