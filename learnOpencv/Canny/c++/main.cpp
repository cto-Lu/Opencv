#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main()
{
    Mat image = imread("../c++/29.jpeg");
    Mat gray, after;
    imshow("initial", image);
    cvtColor(image, gray, COLOR_BGR2GRAY);
    blur(gray, after, Size(3,3));
    Canny(after, after, 50, 130, 3);
    imshow("after", after);
    waitKey(0);
    return 0;
}