#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>

using namespace cv;

int main()
{
    
    Mat image0 = imread("/home/stoair/Opencv/Images/1.png", 4);
    namedWindow("test",WINDOW_NORMAL);
    namedWindow("change1", WINDOW_NORMAL);
    namedWindow("change2", WINDOW_NORMAL);
    namedWindow("change3", WINDOW_NORMAL);
    namedWindow("change4", WINDOW_NORMAL);

    moveWindow("change1",0,0);
    moveWindow("change2",10000,0);
    moveWindow("change3",0,10000);
    moveWindow("change4",10000,10000);

    imshow("test", image0);
    imshow("change1", image0);
    imshow("change2", image0);
    imshow("change3", image0);
    imshow("change4", image0);
    waitKey(0);
    return 0;
}