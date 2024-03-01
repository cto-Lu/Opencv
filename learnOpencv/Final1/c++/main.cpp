#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main()
{
    Mat image = imread("../c++/52.jpg");
    Mat image2 = image.clone();
    int height = image.rows;
    int width = image.cols;
    Mat cir = Mat::zeros(height, width, CV_8UC3);
    imshow("initial", image);
    cvtColor(image, image, COLOR_BGR2GRAY);
    GaussianBlur(image, image, Size(9,9), 2, 2);
    vector<Vec3f> circles;
    HoughCircles(image, circles, HOUGH_GRADIENT, 1, 5, 100, 40);
    cout << "Number of circle is : " << circles.size() << endl;
    cvtColor(image, image, COLOR_GRAY2BGR);
    for(size_t i = 0; i < circles.size(); i++)
    {
        Vec3f cc = circles[i];
        Point center(cvRound(cc[0]), cvRound(cc[1]));
        int radius = cvRound(cc[2]);
        Scalar color1(0, 255, 255);
        circle(image2, center, radius, color1, -1);
    }    
    imshow("change", image2);
    Mat black = Mat::zeros(image.size(), CV_8UC3);
    for (size_t i = 0; i < circles.size(); i++) {
        Vec3f cc = circles[i];
        Point center(cvRound(cc[0]), cvRound(cc[1]));
        int radius = cvRound(cc[2]);
        Scalar color1(255, 255, 255);
        circle(black, center, radius, color1, -1);
    }
    Mat ROI;
    bitwise_and(image, black, ROI);
    imshow("ROI", ROI);
    waitKey(0);
    return 0;
}