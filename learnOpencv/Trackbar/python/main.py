import cv2 as cv

image = cv.imread('../c++/41.jpg')
thresh = 127
maxVal = 255
def updateThreshold(X):
    thresh = cv.getTrackbarPos('Threshold Value', 'image')
    maxVal = cv.getTrackbarPos('Max Value', 'image')
    _, change = cv.threshold(image, thresh, maxVal, cv.THRESH_BINARY)
    cv.imshow('image', change)

cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
cv.createTrackbar('Threshold Value', 'image', thresh, 255, updateThreshold)
cv.createTrackbar('Max Value', 'image', maxVal, 255, updateThreshold)
cv.waitKey(0)
