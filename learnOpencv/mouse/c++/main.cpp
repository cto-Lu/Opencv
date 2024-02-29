#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/opencv.hpp>

using namespace cv;

Mat image;

void onMouse(int event, int x, int y, int flags, void* param)
{
    unsigned char b = image.at<Vec3b>(x, y)[0];
    unsigned char g = image.at<Vec3b>(x, y)[1];
    unsigned char r = image.at<Vec3b>(x, y)[2];
    if(event == EVENT_LBUTTONDOWN)
    {
        printf("the position of the mouse is (%d, %d)\n", x, y);
        printf("the color is (%d, %d, %d)\n\n", b, g, r);
    }
}
int main()
{
    image = imread("../c++/29.jpeg");
    namedWindow("image", WINDOW_AUTOSIZE);
    setMouseCallback("image", onMouse);
    imshow("image", image);
    waitKey(0);
}