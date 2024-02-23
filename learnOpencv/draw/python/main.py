import cv2 as cv
import numpy as np

image = np.zeros((500, 500, 3), dtype = np.uint8)
cv.line(image, (0,0), (100,100), (255,0,0), 3)
cv.rectangle(image, (100,100), (200,200), (0,0,255),3)
cv.circle(image, (200,200), 100,(0,250,0), 3)
cv.imshow("pic", image)
cv.waitKey(0)

