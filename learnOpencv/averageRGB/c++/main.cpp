#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    Mat image0 = imread("/home/stoair/Opencv/Images/1.png", 4);
    Mat image1 = imread("/home/stoair/Opencv/Images/5.png", 5);
    Mat_<Vec3b> image(image1);
    namedWindow("initial", WINDOW_NORMAL);
    namedWindow("change", WINDOW_NORMAL);
    namedWindow("another",WINDOW_NORMAL);
    imshow("initial", image0);

    int height = image0.rows;
    int width = image0.cols;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            unsigned char b = image0.at<Vec3b>(i, j)[0];
            unsigned char g = image0.at<Vec3b>(i, j)[1];
            unsigned char r = image0.at<Vec3b>(i, j)[2];
            unsigned char ave = b/3 + g/3 +r/3;
            image0.at<Vec3b>(i, j)[0] = ave;
            image0.at<Vec3b>(i, j)[1] = ave;
            image0.at<Vec3b>(i, j)[2] = ave;
        }
    }

    int height1 = image1.rows;
    int width1 = image1.cols;
    for(int i = 0; i < height1; i++){
        for(int j = 0; j < width1; j++){
            unsigned char b = image(i, j)[0];
            unsigned char g = image(i, j)[1];
            unsigned char r = image(i, j)[2];
            unsigned char ave = (b+g+r)/3;
            image(i,j) = Vec3b(ave, ave, ave);
        }
    }

    imshow("change", image0);
    imshow("another",image1);
    waitKey(0);
    return 0;
}