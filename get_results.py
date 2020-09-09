from cnn import LetterReader
from image_preprocessor import ImageProcessor
import os
from text_detection import BreakingWords


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


def read_encoding(encoding, map):
    for i in range(len(encoding[0])):
        if encoding[0][i] >= 0.9:
            print(map[i])
            return map[i]


def load_results(model):
    cnn = LetterReader()
    cnn.load(model)

    output = "Word: "

    for i in os.listdir("./temp"):
        encoding = cnn.prediction("./temp/"+i)
        print(encoding)
        try:
            output += read_encoding(encoding, index_map)
        except TypeError:
            print("Letter Skip Detected")

    print(output)


if __name__ == "__main__":

    while True:
        command = input("Type 'train' to train model or 'load' to get results: \n")
        if command == "train":
            setup()
        elif command == "load":
            # getting the main image
            IMAGE_PATH = "dataset/letters/single_prediction/"
            pic = input("Name of image: \n")
            IMAGE_PATH += pic

            # splitting the words in the main image into seperate images
            split = BreakingWords(IMAGE_PATH)
            split.get_binding_box_image()

            # setting up the cnn
            load_results('model.h5')

            # cleaning up the temp folder
            # split.purge_temp()

        elif command == "quit":
            break

