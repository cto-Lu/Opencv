#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main()
{
    Mat image1 = imread("../c++/15.jpg");
    Mat image2 = imread("../c++/16.jpg");
    unsigned char lut[256]; 
    unsigned char lut2[256]; 
    double gamma1 = 0.2;
    double gamma2 = 2;
    for(int i = 0; i < 256; i++)
    {
        lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), gamma1) * 255.0f);
    }
    for(int j = 0; j < 256; j++)
    {
        lut2[j] = saturate_cast<uchar>(pow((float)(j / 255.0), gamma2) * 255.0f);
    }
    const int channels = image1.channels(); 
    const int channels2 = image2.channels(); 
    cout << "channels = " << channels;
    cout << "channels = " << channels2;
    MatIterator_<Vec3b> it,end;
    
    for(it = image1.begin<Vec3b>(), end = image1.end<Vec3b>(); it !=end; it++)
    {
        (*it)[0] = lut[((*it)[0])];  
        (*it)[1] = lut[((*it)[1])];  
        (*it)[2] = lut[((*it)[2])];  
    }
    MatIterator_<Vec3b> it2,end2;
    for(it2 = image2.begin<Vec3b>(), end2 = image2.end<Vec3b>(); it2 !=end2; it2++)
    {
        (*it2)[0] = lut2[((*it2)[0])];  
        (*it2)[1] = lut2[((*it2)[1])];  
        (*it2)[2] = lut2[((*it2)[2])];  
    }
    imshow("pic1",image1);
    imshow("pic2", image2);
    waitKey(0);
    return 0;
}