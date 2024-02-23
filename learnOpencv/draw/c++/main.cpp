#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main()
{
    Point pt1 = Point(0,0);
    Point pt2 = Point(100,100);
    Mat image = Mat::zeros(Size(500, 500), CV_8UC3);
    line(image, pt1, pt2, Scalar(255,0,0), 3);
    rectangle(image, Point(100,100), Point(300, 300), Scalar(0, 255, 0), 3);
    circle(image , Point(300, 300), 200, Scalar(0, 0, 255), 3);
    imshow("pic", image);
    waitKey(0);
    return 0;
}