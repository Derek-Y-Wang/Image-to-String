import cv2
import matplotlib.pyplot as plt
from text_detection import BreakingWords
import pytesseract


# img = cv2.imread("./dataset/letters/single_prediction/hello.png", cv2.COLOR_BGR2GRAY)
# print(img.size)
# h, w, channel = img.shape
# img = cv2.resize(img, dsize=(int(w * 0.2), int(h * 0.2)), interpolation=cv2.INTER_CUBIC)
# plt.imshow(img)
# plt.show()
l = BreakingWords("./dataset/letters/single_prediction/yo.png")
l._process_image()
boxes = pytesseract.image_to_boxes(l.img)
for b in boxes.splitlines():
    b = b.split(' ')
    print(b)
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    rect = cv2.rectangle(l.img, (x-5, y-5), (w + 5,h + 5), (0, 255, 0), 2)

    cv2.imshow("Rectangled", rect)
cv2.waitKey(0)
l.purge_temp()
l.get_binding_box_image()
l.purge_temp()
