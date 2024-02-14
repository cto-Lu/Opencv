# Opencv

## 一、读取图片

### 1.imshow

```c++
Mat imread(const string& filename, intflags=1 );

flags:
enum
{
/* 8bit, color or not */
   CV_LOAD_IMAGE_UNCHANGED  =-1,
/* 8bit, gray */
   CV_LOAD_IMAGE_GRAYSCALE  =0,
/* ?, color */
   CV_LOAD_IMAGE_COLOR      =1,
/* any depth, ? */
   CV_LOAD_IMAGE_ANYDEPTH   =2,
/* ?, any color */
   CV_LOAD_IMAGE_ANYCOLOR   =4
};

Mat image0=imread("dota.jpg",CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);//载入最真实的图像
Mat image1=imread("dota.jpg",0);//载入灰度图
Mat image2=imread("dota.jpg",199);//载入3通道的彩色图像
Mat logo=imread("dota_logo.jpg");//载入3通道的彩色图像
```

- CV_LOAD_IMAGE_UNCHANGED，这个标识在新版本中被废置了，忽略。
- CV_LOAD_IMAGE_ANYDEPTH- 如果取这个标识的话，若载入的图像的深度为16位或者32位，就返回对应深度的图像，否则，就转换为8位图像再返回。
- CV_LOAD_IMAGE_COLOR- 如果取这个标识的话，总是转换图像到彩色一体
- CV_LOAD_IMAGE_GRAYSCALE- 如果取这个标识的话，始终将图像转换成灰度

****

- flags >0返回一个3通道的彩色图像。
- flags =0返回灰度图像。
- flags <0返回包含Alpha通道的加载的图像。

### 2.namedWindow

```c++
void namedWindow(const string& winname,int flags=WINDOW_AUTOSIZE ); 
```

 - WINDOW_NORMAL设置了这个值，用户便可以改变窗口的大小（没有限制）
  - WINDOW_AUTOSIZE如果设置了这个值，窗口大小会自动调整以适应所显示的图像，并且不能手动改变窗口大小。
  - WINDOW_OPENGL 如果设置了这个值的话，窗口创建的时候便会支持OpenGL。

### 3.imshow

```c++
void imshow(const string& winname, InputArray mat);
```

## 二、

