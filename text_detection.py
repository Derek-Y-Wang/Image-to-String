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
            ROI = self.img[y-5:h+5, x-5:w+5]
            cv2.imwrite('./temp/ROI_{}.png'.format(letter_count), ROI)
            letter_count += 1

    def purge_temp(self):
        path = self.temp
        for item in os.listdir(path):
            if item != "placeholder":
                os.remove(path+"/"+item)



# img = cv2.imread("./dataset/letters/single_prediction/hello.png", cv2.COLOR_BGR2GRAY)
# # print(img.size)
# h, w, channel = img.shape
# img = cv2.resize(img, dsize=(int(w * 0.4), int(h * 0.4)), interpolation=cv2.INTER_CUBIC)
# plt.imshow(img)
# plt.show()
# l = BreakingWords("./dataset/letters/single_prediction/yo.png")
# l._process_image()
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



