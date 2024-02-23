import cv2 as cv
import numpy as np

image = cv.imread("../python/15.jpg")
image2 = cv.imread("../python/16.jpg")
gamma = 0.2
gamma2 = 2.0
cv.imshow("ini pic1",image)
cv.imshow("ini pic2",image2)

after = np.power(image/255.0, gamma) * 255
after = np.uint8(after)
after2 = np.power(image2/255.0, gamma2) * 255
after = np.uint8(after)
cv.imshow("pic1",after)
cv.imshow("pic2",after2)
cv.waitKey(0)
