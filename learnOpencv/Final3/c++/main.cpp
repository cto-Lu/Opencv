#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main()
{
    Mat image = imread("./54.jpg");
    Mat initial = image.clone();
    Mat red1, red2, red3, open;
    cvtColor(image, image, COLOR_BGR2HSV);
    Mat image2 = image.clone();
    imshow("image", image);
    inRange(image, Scalar(0, 50, 50), Scalar(10, 255, 255),red1);
    inRange(image2, Scalar(160, 40, 40), Scalar(180, 255, 255), red2);
    addWeighted(red1, 0.5, red2, 0.5, 0,red3);
    Mat element = getStructuringElement(MORPH_RECT,Size(15, 15));
    morphologyEx(red3, open, MORPH_OPEN, element);
    imshow("red", open);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(open, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    drawContours(image, contours, -1, Scalar(0, 255, 0), 1);
    for (size_t i = 0; i < contours.size(); i++) {
        Rect boundRect = boundingRect(contours[i]);
        rectangle(initial, boundRect, Scalar(0, 255, 255), 2);
    }
    imshow("ROI", initial);
    waitKey(0);
}
