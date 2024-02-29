#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main()
{
    Mat image = imread("../c++/32.png");
    int height = image.rows;
    int width = image.cols;
    Mat cir = Mat::zeros(height, width, CV_8UC3);
    imshow("initial", image);
    cvtColor(image, image, COLOR_BGR2GRAY);
    GaussianBlur(image, image, Size(9,9), 2, 2);
    vector<Vec3f> circles;
    HoughCircles(image, circles, HOUGH_GRADIENT, 1, 20, 100, 53);
    cout << "Number of circle is : " << circles.size() << endl;
    for(size_t i = 0; i < circles.size(); i++)
    {
        Vec3f cc = circles[i];
        Point center(cvRound(cc[0]), cvRound(cc[1]));
        int radius = cvRound(cc[2]);
        Scalar color1(rand() % 256, rand() % 256, rand() % 256);
        circle(cir, center, radius, color1, -1);
        Scalar color2(rand() % 256, rand() % 256, rand() % 256);
        circle(cir, center, 3, color2, -1);
    }    
    Mat copy = cir.clone();
    // threshold(cir, cir, 0,255,THRESH_BINARY);
    cvtColor(cir, cir, COLOR_BGR2GRAY);
    Mat labels, stats, centroids;
    int num = connectedComponentsWithStats(cir, labels, stats, centroids);
    for (int i = 1; i < num; ++i) {
		
			Rect boundingRect(
				stats.at<int>(i, CC_STAT_LEFT),
				stats.at<int>(i, CC_STAT_TOP),
				stats.at<int>(i, CC_STAT_WIDTH),
				stats.at<int>(i, CC_STAT_HEIGHT));
			rectangle(copy, boundingRect, Scalar(0, 255, 0), 2);
		
	}
    imshow("after", copy);
    waitKey(0);
    return 0;
}