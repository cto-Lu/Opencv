#include "yolov8.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <chrono>

#define USE_CUDA false //use opencv-cuda

using namespace std;
using namespace cv;
using namespace dnn;
using namespace std::chrono;

int main()
{
    string img_path = "./broke_0011.jpg";
    string model_path3 = "./best.onnx";
    Mat img = imread(img_path);

    vector<Scalar> color;
    srand(time(0));
    for (int i = 0; i < 80; i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(Scalar(b, g, r));
    }

    Yolov8 yolov8;
    Net net3;
    Mat img3 = img.clone();

    bool isOK = yolov8.readModel(net3, model_path3, USE_CUDA);
    if (isOK) {
        cout << "read net ok!" << endl;
    } else {
        cout << "read onnx model failed!";
        return -1;
    }

    steady_clock::time_point start_time = steady_clock::now();

    vector<Detection> result3 = yolov8.Detect(img3, net3);

    steady_clock::time_point end_time = steady_clock::now();
    duration<double> detection_time = duration_cast<duration<double>>(end_time - start_time);
    cout << "Detection time: " << detection_time.count() << " seconds" << endl;

    yolov8.drawPred(img3, result3, color);
    Mat dst = img3({ 0, 0, img.cols, img.rows });
    cv::imshow("aaa", dst);
    imwrite("./yolov8.jpg", dst);
    cv::waitKey(0);

    return 0;
}

