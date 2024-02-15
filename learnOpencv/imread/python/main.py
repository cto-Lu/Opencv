import cv2 as cv
image0 = cv.imread("/home/stoair/Opencv/Images/1.png", 4);
cv.namedWindow("test",cv.WINDOW_NORMAL);
cv.namedWindow("change1", cv.WINDOW_NORMAL);
cv.namedWindow("change2", cv.WINDOW_NORMAL);
cv.namedWindow("change3", cv.WINDOW_NORMAL);
cv.namedWindow("change4", cv.WINDOW_NORMAL);

cv.moveWindow("change1",0,0);
cv.moveWindow("change2",10000,0);
cv.moveWindow("change3",0,10000);
cv.moveWindow("change4",10000,10000);

cv.imshow("test", image0);
cv.imshow("change1", image0);
cv.imshow("change2", image0);
cv.imshow("change3", image0);
cv.imshow("change4", image0);
cv.waitKey(0);
