import cv2 as cv

def onMouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        (b, g, r) = image[x, y]
        print("the point (x,y) of the mouse is (%d, %d)" % (x, y),)
        print("the color is (%d, %d, %d)" % (b, g, r))



image = cv.imread('../python/29.jpeg')
cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
cv.setMouseCallback('image', onMouse)
cv.imshow('image', image)
cv.waitKey(0)