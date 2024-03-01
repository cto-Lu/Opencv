#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main()
{
    Mat image = imread("./53.jpg");
    int height = image.rows;
    int width = image.cols;
    Mat black = Mat::zeros(height, width, CV_8UC3);
    rectangle(black, Point(178, 173), Point(270, 261), Scalar(255, 255, 255), -1);
    Mat ROI, gray;
    bitwise_and(image, black, ROI);
    imshow("ROI", ROI);
    cvtColor(ROI, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 1);
    double ret = threshold(gray, gray, 168, 255, THRESH_BINARY);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    drawContours(image, contours, -1, Scalar(0, 255, 0), 1);
    for (size_t i = 0; i < contours.size(); i++) {
        Rect boundRect = boundingRect(contours[i]);
        rectangle(image, boundRect, Scalar(0, 255, 255), 2);
    }
    imshow("image", image);
    waitKey(0);
    return 0;
}