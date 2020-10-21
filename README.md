# Image-to-String
This project is built with a convolutional neural network developed using Tensorflow and Keras and trained with the NIST characters dataset.
https://www.nist.gov/srd/nist-special-database-19 

This model developed to perform the image to string action. pytesseract's OCR library was used inorder to detect each individual letter in a string, 
which is then sent to a neural network inorder to be processed and converted into a string.

NOTE: Results may not be accurate.

# How to use it.
Step 1: First make sure to save the image string into the 
``` dataset/letters/single_prediction```
folder.

Step 2: Run the ```get_results.py``` file and make type ```load``` in the terminal when prompted ```Type 'train' to train model or 'load' to get results: ```

Step 3: Type the image name you wish to process from ``` dataset/letters/single_prediction```


