question:

**pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib**

# Opencv

## 一、读取图片

### (1).imshow

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

### (2).namedWindow

```c++
void namedWindow(const string& winname,int flags=WINDOW_AUTOSIZE ); 
```

 - WINDOW_NORMAL设置了这个值，用户便可以改变窗口的大小（没有限制）
  - WINDOW_AUTOSIZE如果设置了这个值，窗口大小会自动调整以适应所显示的图像，并且不能手动改变窗口大小。
  - WINDOW_OPENGL 如果设置了这个值的话，窗口创建的时候便会支持OpenGL。

### (3).imshow

```c++
void imshow(const string& winname, InputArray mat);
```

### (4).效果图

c++

<img src="../Opencv/Images/4.png" >

python

<img src="../Opencv/Images/5.png" >

## 二、像素操作

### (1).访问像素

#### 1. at()

```c++
image.at<uchar>(j,i)= value;  //单通道
image.at<cv::Vec3b>(j,i)[channel]= value;  //三通道
image.at<cv::Vec3b>(j,i) = cv::Vec3b(a,b,c);
```

#### 2.Mat_

```c
cv::Mat_<uchar> image(image1);
image(20,30) = value;
```

### (2).遍历像素

#### 1.指针遍历

```c++
uchar *data = image.ptr<uchar>(i);  //ptr()返回行的地址
```

```c++
for (int i = 0; i < height; i++) {
    cv::Vec3b* row = image.ptr<cv::Vec3b>(i);
    for (int j = 0; j < width; j++) {
        cv::Vec3b& pixel = row[j];//Vec3b&直接操作图像中的像素值，而不需要创建新的对象
        std::cout << "Pixel at (" << i << "," << j << "): "
                  << "B=" << (int)pixel[0] << " "
                  << "G=" << (int)pixel[1] << " "
                  << "R=" << (int)pixel[2] << std::endl;
    }
}
```

#### 2.迭代器遍历

```c++
cv::MatIterator_ <cv::Vec3b> it;
或者
cv::Mat_<cv::Vec3b>::iterator it;
```

```c++
cv::MatIterator_<cv::Vec3b> it, end;
for (it = image.begin<cv::Vec3b>(), end = image.end<cv::Vec3b>(); it != end; ++it) {
    
    cv::Vec3b& pixel = *it;
    
    pixel[0] = 255; 
    pixel[1] = 0; 
    pixel[2] = 0; 
}
```

python

<img src="../Opencv/Images/6.png" >

c++

<img src="../Opencv/Images/7.png" >

### (3).threshold

```c++
double cv::threshold(src, OutputArray, thresh, maxval, type)
```

![image-20240218155810216](../Opencv/Images/8.png)

c++:

![image-20240218162057595](../Opencv/Images/9.png)

python:

![image-20240218163235173](../Opencv/Images/10.png)

### (4).通道分离

#### 1.split

```c++
C++: void split(const Mat& src, Mat*mvbegin);
C++: void split(InputArray m,OutputArrayOfArrays mv);
```

#### 2.merge

```c++
C++: void merge(const Mat* mv, size_tcount, OutputArray dst)
C++: void merge(InputArrayOfArrays mv,OutputArray dst)
```

c++

![image-20240218171735253](../Opencv/Images/11.png)

python

![image-20240218173748516](../Opencv/Images/12.png)

### (5)Gamma矫正

Gamma校正是对输入图像灰度值进行的非线性操作，使输出图像灰度值与输入图像灰度值呈指数关系。Gamma`矫正用于调整图像的亮度和对比度`。Gamma矫正可以改变图像的灰度值分布，使图像在显示时看起来更加自然和逼真。通常情况下，人眼对亮度的感知是非线性的，因此使用Gamma矫正可以更好地模拟人眼的感知特性。
$$
V_{out}=AV_{in}^\gamma
$$
γ的值决定了输入图像和输出图像之间的灰度映射方式，即决定了是增强低灰度值区域还是增高灰度值区域。
γ>1时，图像的高灰度区域对比度得到增强，直观效果是一幅偏亮的图变暗了下来。
γ<1时，图像的低灰度区域对比度得到增强，直观效果是一幅偏暗的图变亮了起来。

python

![image-20240218214951987](../Opencv/Images/17.png)

c++

![image-20240218232830142](../Opencv/Images/18.png)

### (6).深浅拷贝

**浅拷贝是指当图像之间进行赋值**时，图像数据并未发生复制，而是两个对象都指向同一块内存块。 

**深拷贝是指新创建的图像拥有原始图像的崭新拷贝**

c++

![image-20240218195610907](../Opencv/Images/13.png)

python

![image-20240218200910278](../Opencv/Images/14.png)


## 三、基本绘图

### (1).line

```c++
void cv::line(InputOutputArray img,Point pt1, Point pt2, const Scalar & color, int  thickness = 1, int  lineType = LINE_8, int  shift = 0)
```

| img       | Image.                                                       |
| --------- | ------------------------------------------------------------ |
| pt1       | First point of the line segment.                             |
| pt2       | Second point of the line segment.                            |
| color     | Line color.                                                  |
| thickness | Line thickness.                                              |
| lineType  | Type of the line. See [LineTypes](https://docs.opencv.org/4.0.0/d6/d6e/group__imgproc__draw.html#gaf076ef45de481ac96e0ab3dc2c29a777). |
| shift     | Number of fractional bits in the point coordinates.          |

### (2).rectangle

```c++
void cv::rectangle(InputOutputArray img, Point pt1, Point pt2, const Scalar & color, int  thickness = 1,int  lineType = LINE_8, int  shift = 0)
 
void cv::rectangle(InputOutputArray img, Rect rec, const Scalar & color, int  thickness = 1,int  lineType = LINE_8, int  shift = 0)            
```

### (3).circle

```c++
void cv::circle(InputOutputArray img, Point center,  int  radius, const Scalar & color, int  thickness = 1, int  lineType = LINE_8, int  shift = 0)
```

python

![image-20240223141111667](../Opencv/Images/22.png)

c++

![image-20240223163950444](../Opencv/Images/23.png)

## 四、图像处理

### (1).颜色空间

#### 1.意义

- RGB 颜色空间利用三个颜色分量的线性组合来表示颜色，任何颜色都与这三个分量有关，而且这三个分量是高度相关的，所以连续变换颜色时并不直观，想对图像的颜色进行调整需要更改这三个分量才行。

- 自然环境下获取的图像容易受自然光照、遮挡和阴影等情况的影响，即对亮度比较敏感。而 RGB 颜色空间的三个分量都与亮度密切相关，即只要亮度改变，三个分量都会随之相应地改变，而没有一种更直观的方式来表达。

- 在图像处理中使用较多的是 HSV 颜色空间，它比 RGB 更接近人们对彩色的感知经验。非常直观地表达颜色的色调、鲜艳程度和明暗程度，方便进行颜色的对比。

H（色调/hue）  |

S（饱和度/saturation）  |

V（明度/Value）  |

![image-20240221214232092](../Opencv/Images/19.png)

#### 2.cvtColor()

```c++
void cv::cvtColor(InputArray src, OutputArray dst, int code, int dstCn=0)
```

- src：输入图像，可以是Mat类型的图像或者其他支持的图像数据结构。
- dst：输出图像，用于存储转换后的图像。
- code：颜色空间转换的代码，例如CV_BGR2GRAY表示将BGR颜色空间转换为灰度图像。
- dstCn：输出图像的通道数，如果为0，则自动根据code参数确定通道数。

#### 3.inRange()

```c++
void inRange(InputArray src, InputArray lowerb,InputArray upperb, OutputArray dst);
void inRange(image, Scalar(hmin,smin,vmin), Scalar(hmax,smax,vmax), image);
//typedef Vec<double, 4> Scalar;
```

python:

![image-20240222035354360](../Opencv/Images/20.png)

c++:

![image-20240222040750118](../Opencv/Images/21.png)

#### 4.适应光线

光线较暗 -> 暗色调 ； 增加饱和度S ；减小亮度V

光线较亮 -> 亮色调 ； 减小饱和度S ；增大亮度V

### (2).形态操作

#### 1.腐蚀

腐蚀的基本概念就像土壤侵蚀一样，只侵蚀前景对象的边界（总是尽量保持前景为白色）。那它有什么作用呢？内核在图像中滑动（如二维卷积）。只有当内核下的所有像素都为 1 时，原始图像中的像素（1 或 0）才会被视为 1，否则会被侵蚀（变为零）。

```c++
C++: void erode(
	InputArray src,
	OutputArray dst,
	InputArray kernel,
	Point anchor=Point(-1,-1),
	int iterations=1,
	int borderType=BORDER_CONSTANT,
	const Scalar& borderValue=morphologyDefaultBorderValue()
 );	
```

```c++
 int g_nStructElementSize = 3; //结构元素(内核矩阵)的尺寸
 
//获取自定义核
Mat element = getStructuringElement(MORPH_RECT,
	Size(2*g_nStructElementSize+1,2*g_nStructElementSize+1),
	Point( g_nStructElementSize, g_nStructElementSize ));
```

#### 2.膨胀

它与腐蚀正好相反。这里，如果内核下至少有一个像素为“1”，则像素元素为“1”。所以它会增加图像中的白色区域，或者增加前景对象的大小。通常情况下，在去除噪音的情况下，腐蚀后会膨胀。因为，腐蚀消除了白噪声，但它也缩小了我们的对象。所以我们扩大它。由于噪音消失了，它们不会再回来，但我们的目标区域会增加到腐蚀之前的状态。它还可用于连接对象的断开部分。

```c++
C++: void dilate(
	InputArray src,
	OutputArray dst,
	InputArray kernel,
	Point anchor=Point(-1,-1),
	int iterations=1,
	int borderType=BORDER_CONSTANT,
	const Scalar& borderValue=morphologyDefaultBorderValue() 
);
```

#### 3.开/闭运算

- 开运算（Opening Operation），其实就是先腐蚀后膨胀的过程。开运算可以用来消除小物体、在纤细点处分离物体、平滑较大物体的边界的同时并不明显改变其面积。

- 先膨胀后腐蚀的过程称为闭运算(Closing Operation)，闭运算能够排除小型黑洞(黑色区域)。

```c++
C++: void morphologyEx(
InputArray src,
OutputArray dst,
int op,
InputArraykernel,
Pointanchor=Point(-1,-1),
intiterations=1,
intborderType=BORDER_CONSTANT,
constScalar& borderValue=morphologyDefaultBorderValue() 
);
```

第三个参数，int类型的op，表示形态学运算的类型，可以是如下之一的标识符：

- MORPH_OPEN – 开运算（Opening operation）
- MORPH_CLOSE – 闭运算（Closing operation）
- MORPH_GRADIENT -形态学梯度（Morphological gradient）
- MORPH_TOPHAT - “顶帽”（“Top hat”）
- MORPH_BLACKHAT - “黑帽”（“Black hat“）
- MORPH_ERODE-"腐蚀"
- MORPH_DILATE-"膨胀"

c++

![image-20240223222254511](../Opencv/Images/25.png)

python

![image-20240223231354329](../Opencv/Images/26.png)

#### 4.error

**problem** : /../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found

**solve** : 系统环境下 /usr/lib/x86_64-linux-gnu/libstdc++.so.6 文件含有GLIBCXX_3.4.30版本，而anaconda环境下libstdc++.so.6文件含有的最高版本为GLIBCXX_3.4.29，因此有了前面的报错。

```
rm libstdc++.so 
rm libstdc++.so.6
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.32 libstdc++.so
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.32 libstdc++.so.6
```

### (3).Ganny边缘检测

