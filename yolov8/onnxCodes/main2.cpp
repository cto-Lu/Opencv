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
    VideoCapture cap(0); // 打开默认摄像头
    if (!cap.isOpened())
    {
        cerr << "Error opening the camera" << endl;
        return -1;
    }

    string model_path3 = "./best.onnx";

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

    bool isOK = yolov8.readModel(net3, model_path3, USE_CUDA);
    if (!isOK) {
        cout << "Failed to read the ONNX model!" << endl;
        return -1;
    }

    while (true)
    {
        Mat frame;
        cap >> frame; // 从摄像头捕获帧

        steady_clock::time_point start_time = steady_clock::now();

        vector<Detection> result3 = yolov8.Detect(frame, net3);

        steady_clock::time_point end_time = steady_clock::now();
        duration<double> detection_time = duration_cast<duration<double>>(end_time - start_time);
        cout << "Detection time: " << detection_time.count() << " seconds" << endl;

        yolov8.drawPred(frame, result3, color);
        imshow("Object Detection", frame);

        if (waitKey(1) == 27) // 按下ESC键退出循环
            break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
