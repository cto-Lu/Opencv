import cv2 as cv
import numpy as np

image = cv.imread("../python/52.jpg")
image2 = cv.imread("./52.jpg")
height = image.shape[0]
width = image.shape[1]
cir = np.zeros((height, width, 3), np.uint8)
cv.imshow("initial", image)

gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray_image = cv.GaussianBlur(gray_image, (9, 9), 2, 2)
circles = cv.HoughCircles(gray_image, cv.HOUGH_GRADIENT, 1, 5, param1=100, param2=40)
print("Number of circles: ", circles.shape[1])
circles = np.uint16(circles)
for i in circles[0, :]:
    center = (i[0], i[1])
    radius = i[2]
    color = (0, 255, 255)
    cv.circle(image, center, radius, color, -1)
cv.imshow("circles", image)
for i in circles[0, :]:
    center = (i[0], i[1])
    radius = i[2]
    color = (255, 255, 255)
    cv.circle(cir, center, radius, color, -1)
ROI = cv.bitwise_and(image2, cir)
cv.imshow("ROI", ROI)
cv.waitKey(0)