import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "./pytesseract\\Tesseract-OCR\\tesseract.exe"
img = cv2.imread("dataset/letters/single_prediction/paint_BOOM7.PNG", cv2.IMREAD_GRAYSCALE)
# print(pytesseract.image_to_string(img))

### Detecting Characters
hImg, wImg = img.shape
boxes = pytesseract.image_to_boxes(img)
letter_count = 0

for b in boxes.splitlines():
    b = b.split(' ')
    b[0] = letter_count
    letter_count += 1
    print(b)
    # print(type(b))
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(img, (x, hImg-y), (w, hImg-h), (0, 0, 255), 2)
    ROI = img[w:hImg-h,  x:hImg-y]
    cv2.imwrite('./temp/ROI.png', ROI)

    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)


# print(pytesseract.image_to_boxes(img))
cv2.imshow('Result', img)
# cv2.imshow("l", img[])
cv2.waitKey(0)

class SeperateCharacters:

    def __init__(self, image):
        try:
            pytesseract.pytesseract.tesseract_cmd = "./pytesseract\\Tesseract-OCR\\tesseract.exe"
            self.img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        except:
            print("Missing pytesseract")


    def get_letters(self):
        hImg, wImg = self.img.shape
        boxes = pytesseract.image_to_boxes(self.img)
        letter_count = 0

        for b in boxes.splitlines():
            b = b.split(' ')
            b[0] = letter_count
            letter_count += 1
            print(b)
            # print(type(b))
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (0, 0, 255), 2)
            # ROI = img[w:hImg - h, x:hImg - y]
            # cv2.imwrite('./temp/ROI.png', ROI)

