import tensorflow as tf
import numpy as np
import os
from keras.preprocessing import image
from cnn import LetterReader


L = LetterReader()
L.load('model.h5')
files = os.listdir("dataset/letters/test_set/")
index_map = dict()
for n in range(len(os.listdir("dataset/letters/test_set/"))):
    index_map[n] = files[n]

encoding = L.prediction("dataset/letters/single_prediction/ROI_3.PNG")

print(encoding)
for i in range(len(encoding[0])):
    if encoding[0][i] >= 0.9:
        print(index_map[i])
