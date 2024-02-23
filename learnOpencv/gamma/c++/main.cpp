#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/opencv.hpp>
#include<cmath>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    Mat image1 = imread("../c++/15.jpg");
    Mat image2 = imread("../c++/16.jpg");
    double gamma1 = 0.2;
    double gamma2 = 2;
    image1.convertTo(image1, CV_64F);
    image2.convertTo(image2, CV_64F);
    image1 = image1 / 255.0;
    image2 = image2 / 255.0;
    pow(image1, gamma1, image1);
    pow(image2, gamma2, image2);
    image1 *= 255.0;
    image2 *= 255.0;
    image1.convertTo(image1, CV_8U);
    image2.convertTo(image2, CV_8U);
    imshow("img1", image1);
    imshow("img2", image2);
    waitKey(0);
    return 0;
}