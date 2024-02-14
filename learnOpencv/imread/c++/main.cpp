#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace cv;

int main()
{
    Mat image0 = imread("/home/stoair/Opencv/Images/1.png", 4);
    //namedWindow("bug");
    //namedWindow("change", WINDOW_NORMAL);
    moveWindow("p1",1024,345);
    //imshow("bug", image0);
    //imshow("change", image0);
    imshow("p1", image0);
    waitKey(0);
    return 0;
}