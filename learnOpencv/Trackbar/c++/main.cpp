#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat image;
int thresh = 127;
int maxval = 255;

void updateThreshold(int, void*)
{
    Mat change;
    threshold(image, change, thresh, maxval, THRESH_BINARY);
    imshow("image", change);
    setTrackbarPos("Threshold Value", "image", thresh);
    setTrackbarPos("Max Value", "Modified", maxval);
}
int main()
{
    image = imread("../c++/41.jpg");
    namedWindow("image", WINDOW_AUTOSIZE);
    createTrackbar("Threshold Value", "image", &thresh, 255, updateThreshold);
    createTrackbar("Max Value", "image", &maxval, 255, updateThreshold);
    waitKey(0);
    return 0;
}

