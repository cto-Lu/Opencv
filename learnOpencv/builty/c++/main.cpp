#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main(){
	Mat image = imread("../c++/41.jpg");
    Mat image2 = imread("../c++/40.jpg");
	imshow("initial", image);
    imshow("initial2", image2);
	Mat change, change2;
	bilateralFilter(image, change, 20, 100, 5);
    bilateralFilter(image2, change2, 20, 100, 5);
	imshow("change1", change);
    imshow("change2", change2);
	waitKey(0);
	return 0;
}
