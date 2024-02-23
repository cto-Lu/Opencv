#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main()
{
    Mat image = imread("/home/stoair/Opencv/Images/1.png", 4);
    vector<Mat> channels;
    namedWindow("blue", WINDOW_NORMAL);
    namedWindow("green", WINDOW_NORMAL);
    namedWindow("red",WINDOW_NORMAL);    
    split(image, channels);
    imshow("blue",channels[0]);
    imshow("green",channels[1]);
    imshow("red",channels[2]);
    waitKey(0);
    return 0;
}