#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main()
{
    Mat image = imread("../c++/24.png");
    Mat out1,out2,open,close;
    Mat element = getStructuringElement(MORPH_RECT,Size(15, 15));
    dilate(image, out1, element);
    erode(image, out2, element);
    morphologyEx(image, open, MORPH_OPEN, element);
    morphologyEx(image, close, MORPH_CLOSE, element);
    imshow("initial", image);
    imshow("dilate", out1);
    imshow("erode", out2);
    imshow("open", open);
    imshow("close", close);
    waitKey(0);
    return 0;
}