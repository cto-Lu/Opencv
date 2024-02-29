import cv2 as cv

image = cv.imread("../python/29.jpeg")
cv.imshow("Original", image)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blur = cv.blur(image, (3,3))
canny = cv.Canny(image, 200, 200, 3)
cv.imshow("Canny", canny)
cv.waitKey(0)

