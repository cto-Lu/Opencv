import cv2 as cv
import numpy as np
import copy
img = cv.imread('./35.png')
img2 = copy.deepcopy(img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE )
cv.drawContours(img, contours, -1, (0,255,0), 1)
cv.imshow('img', img)
cv.imshow('thresh', thresh)
for cnt in contours:
    hull = cv.convexHull(cnt,returnPoints=True)
    cv.polylines(img2, [hull],True,(0,0,255), 1)
num = 0
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        num += 1

print("外轮廓数量：", num)
cv.imshow('hull', img2)
cv.waitKey(0)