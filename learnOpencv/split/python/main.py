import cv2 as cv
image = cv.imread("/home/stoair/Opencv/Images/1.png", 4)
cv.namedWindow("blue",cv.WINDOW_NORMAL);
cv.namedWindow("green",cv.WINDOW_NORMAL);
cv.namedWindow("red",cv.WINDOW_NORMAL);

b,g,r = cv.split(image)
cv.imshow("blue",b)
cv.imshow("green",g)
cv.imshow("red",r)
cv.waitKey(0)
