import cv2
import pytesseract
import os
from cnn import LetterReader


class BreakingWords:

    def __init__(self, image):
        try:
            pytesseract.pytesseract.tesseract_cmd = "./pytesseract\\Tesseract-OCR\\tesseract.exe"
            self.img = cv2.imread(image, cv2.COLOR_BGR2RGB)
            self.temp = "./temp"
        except:
            print("Missing pytesseract")

    def get_binding_box_image(self):
        boxes = pytesseract.image_to_boxes(self.img)
        letter_count = 0

        for b in boxes.splitlines():
            b = b.split(' ')
            b[0] = letter_count
            letter_count += 1
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            ROI = self.img[y-5:h+5, x-5:w+5]
            cv2.imwrite('./temp/ROI_{}.png'.format(letter_count), ROI)
            letter_count += 1

    def purge_temp(self):
        path = self.temp
        for item in os.listdir(path):
            os.remove(path+"/"+item)



# l = BreakingWords("./dataset/letters/single_prediction/hello.png")
# boxes = pytesseract.image_to_boxes(l.img)
# for b in boxes.splitlines():
#     b = b.split(' ')
#     print(b)
#     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#     rect = cv2.rectangle(l.img, (x-5, y-5), (w + 5,h + 5), (0, 255, 0), 2)
#
#     cv2.imshow("Rectangled", rect)
# cv2.waitKey(0)
# l.purge_temp()
# l.get_binding_box_image()
# l.purge_temp()



