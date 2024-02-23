#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    int x = 200;
    int y = 100;
    unsigned char b, g, r;
    Mat image0 = imread("/home/stoair/Opencv/Images/1.png", 4);
    Mat image1 = imread("/home/stoair/Opencv/Images/1.png", 4);
    namedWindow("initial", WINDOW_NORMAL);
    namedWindow("change", WINDOW_NORMAL);
    namedWindow("another",WINDOW_NORMAL);
    imshow("initial", image0);

    int height = image0.rows;
    int width = image0.cols;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            b = image0.at<Vec3b>(i, j)[0];
            g = image0.at<Vec3b>(i, j)[1];
            r = image0.at<Vec3b>(i, j)[2];
            unsigned char ave = b/3 + g/3 +r/3;
            if(ave > x)
                ave = 255;
            else
                ave = 0;
            image0.at<Vec3b>(i, j)= Vec3b(ave, ave, ave);
        }
    }
    imshow("change", image0);

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            b = image1.at<Vec3b>(i, j)[0];
            g = image1.at<Vec3b>(i, j)[1];
            r = image1.at<Vec3b>(i, j)[2];
            unsigned char ave = b/3 + g/3 +r/3;
            if(ave > y)
                ave = 255;
            else
                ave = 0;
            image1.at<Vec3b>(i, j)= Vec3b(ave, ave, ave);
        }
    }

    imshow("another",image1);
    waitKey(0);
    return 0;
}