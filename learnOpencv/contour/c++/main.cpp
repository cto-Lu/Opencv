#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main()
{
    Mat image = imread("./35.png");
    Mat image2 = image.clone();
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    double ret = threshold(gray, gray, 127, 255, THRESH_BINARY_INV);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(gray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    drawContours(image, contours, -1, Scalar(0,255,0),1);
    imshow("image", image);
    imshow("gray", gray);
    int num = 0;
    for (size_t i = 0; i < contours.size(); i++)
    {
        vector<Point> cnt = contours[i];
        vector<Point> hull;
        convexHull(cnt, hull, true);
        std::vector<std::vector<Point>> hulls = {hull};
        polylines(image2, hulls, true, Scalar(0, 0, 255), 1);
        if(hierarchy[i][3] == -1)
            num++;
    }
    cout << num << endl;
    imshow("image2", image2);
    waitKey(0);
}