#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main()
{
    Mat image = imread("../c++/19.png");
    Mat blue;
    Mat red;
    cvtColor(image, image, COLOR_BGR2HSV);
    imshow("pic", image);
    inRange(image, Scalar(100, 50, 50),Scalar(130, 255, 255), blue);
    inRange(image, Scalar(0, 50, 50), Scalar(10, 255, 255),red);
    imshow("blue", blue);
    imshow("red", red);
    waitKey(0);
    return 0;
}