import cv2
import pytesseract
import os
import matplotlib.pyplot as plt


class BreakingWords:

    def __init__(self, image):
        try:
            pytesseract.pytesseract.tesseract_cmd = "./pytesseract\\Tesseract-OCR\\tesseract.exe"
            self.img = cv2.imread(image, cv2.COLOR_BGR2GRAY)
            self.temp = "./temp"
        except:
            print("Missing pytesseract")

    def _process_image(self):
        h, w, channel = self.img.shape
        self.img = cv2.resize(self.img, dsize=(int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_AREA)
        return self.img

    def get_binding_box_image(self):
        self._process_image()
        boxes = pytesseract.image_to_boxes(self.img)
        letter_count = 0
        if boxes is None:
            boxes = pytesseract.image_to_boxes(self.img, config="--psm 10")
        for b in boxes.splitlines():
            b = b.split(' ')
            b[0] = letter_count
            letter_count += 1
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            if y-5 >= 0 and h+5 <= len(self.img) and x-5 >= 0 and w+5 <= len(self.img[0]):
                ROI = self.img[y-5:h+5, x-5:w+5]
            else:
                ROI = self.img[y:h, x:w]
            # ROI = self.img[y - 5:h + 5, x - 5:w + 5]
            cv2.imwrite('./temp/ROI_{}.png'.format(letter_count), ROI)
            letter_count += 1

    def purge_temp(self):
        path = self.temp
        for item in os.listdir(path):
            if item != "placeholder":
                os.remove(path+"/"+item)







