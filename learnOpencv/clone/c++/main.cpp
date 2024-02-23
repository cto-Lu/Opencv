#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
    int x = 200;
    int y = 100;
    unsigned char b, g, r;
    Mat image0 = imread("/home/stoair/Opencv/Images/1.png", 4);
    Mat image1 = imread("/home/stoair/Opencv/Images/1.png", 4);
    Mat img0 = image0;
    
    int height = image0.rows;
    int width = image0.cols;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            b = img0.at<Vec3b>(i, j)[0];
            g = img0.at<Vec3b>(i, j)[1];
            r = img0.at<Vec3b>(i, j)[2];
            unsigned char ave = b/3 + g/3 +r/3;
            if(ave > x)
                ave = 255;
            else
                ave = 0;
            img0.at<Vec3b>(i, j)= Vec3b(ave, ave, ave);
        }
    }
    imshow("shallow", image0);

    Mat img1 = image1.clone();
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            b = img1.at<Vec3b>(i, j)[0];
            g = img1.at<Vec3b>(i, j)[1];
            r = img1.at<Vec3b>(i, j)[2];
            unsigned char ave = b/3 + g/3 +r/3;
            if(ave > y)
                ave = 255;
            else
                ave = 0;
            img1.at<Vec3b>(i, j)= Vec3b(ave, ave, ave);
        }
    }
    imshow("deep", image1);
    waitKey(0);
    return 0;
}