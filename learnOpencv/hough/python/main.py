import cv2 as cv
import numpy as np

image = cv.imread("../c++/32.png")
height = image.shape[0]
width = image.shape[1]
cir = np.zeros((height, width, 3), np.uint8)
cv.imshow("initial", image)

gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray_image = cv.GaussianBlur(gray_image, (9, 9), 2, 2)
circles = cv.HoughCircles(gray_image, cv.HOUGH_GRADIENT, 1, 20, param1=100, param2=53)
print("Number of circles: ", circles.shape[1])
circles = np.uint16(circles)
for i in circles[0, :]:
    center = (i[0], i[1])
    radius = i[2]
    color1 = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    cv.circle(cir, center, radius, color1, -1)
    color2 = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    cv.circle(cir, center, 3, color2, -1)

copy = cir.copy()
cir = cv.cvtColor(cir, cv.COLOR_BGR2GRAY)
num, labels, stats, centroids = cv.connectedComponentsWithStats(cir)
for i in range(1, num):
    left = stats[i, cv.CC_STAT_LEFT]
    top = stats[i, cv.CC_STAT_TOP]
    width = stats[i, cv.CC_STAT_WIDTH]
    height = stats[i, cv.CC_STAT_HEIGHT]
    cv.rectangle(copy, (left, top), (left+width, top+height), (0, 255, 0), 2)
cv.imshow("after", copy)
cv.waitKey(0)
