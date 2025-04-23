#include <opencv2/opencv.hpp>
#include <iostream>
#include<string>
#include <vector>
#include <algorithm>
#include <stdexcept>

using namespace std;
using namespace cv;

void myblur(const Mat& src, Mat& dst, Size ksize);//均值滤波
void addSaltNoise(const Mat& srcImage, Mat& dstImage, int n);//加入椒盐噪声
void addGaussianNoise(Mat& srcImag, Mat& dstImage, double mu, double sigma);//加入高斯噪声(mu:期望,sigma:方差)
void median_filter(const Mat& src, Mat& dst, int ksize);//中值滤波
void sharpen_FirstOrder(const Mat& src, Mat& dst);//一阶微分算子锐化
void Sharpen_laplace(cv::Mat& inputImage, cv::Mat& outputImage);//二阶拉普拉斯算子锐化

Mat dstImg_blur_SaltNoise;//均值滤波滤椒盐噪声后图像
Mat dstImg_blur_GaussianNoise;//均值滤波滤高斯噪声后图像
Mat dstImg_median_SaltNoise;//中值滤波滤椒盐噪声后图像
Mat dstImg_median_GaussianNoise;//中值滤波滤椒盐噪声后图像
Mat dstImg_SaltNoise;//加入椒盐噪声后图像
Mat dstImg_GaussianNoise;//加入高斯噪声图像
Mat dstImg_sharpen_FirstOrder;//一阶微分锐化后图像
Mat dstImg_sharpen_SecondOrder;//二阶微分锐化后图像

int main()
{
    //OpenCV版本号
    cout << "OpenCV_Version: " << CV_VERSION << endl;
    //读取图片
    Mat Img = imread("D:/OpenCV/Data set/scenery/tower.jpg");
    //读取灰度图
    Mat Image_gray = imread("D:/OpenCV/Data set/Heart_stealer/xiugou.jpg", 0);
    if ((!Img.data) || (!Image_gray.data))
    {
        cout << "读入图片失败" << endl;
        return -1;
    }
    //读取图片行数和列数
    int Rowsnum = Img.rows;//高
    int Colsnum = Img.cols;//宽

    namedWindow("原图像", WINDOW_AUTOSIZE);
    imshow("原图像", Img);//显示原图像
    addSaltNoise(Img, dstImg_SaltNoise, 50000);
    imshow("加入椒盐噪声图像", dstImg_SaltNoise);//显示椒盐图像
    addGaussianNoise(Img, dstImg_GaussianNoise,0,1.5);
    imshow("加入高斯噪声图像", dstImg_GaussianNoise);//显示高斯图像

    /*
    //均值滤波
    dstImg_blur_SaltNoise =Mat::zeros(dstImg_SaltNoise.size(), dstImg_SaltNoise.type());
    myblur(dstImg_SaltNoise, dstImg_blur_SaltNoise, Size(3, 3));
    imshow("椒盐噪声均值滤波后图像", dstImg_blur_SaltNoise);//显示原图像
    //
    dstImg_blur_GaussianNoise = Mat::zeros(dstImg_GaussianNoise.size(), dstImg_GaussianNoise.type());
    myblur(dstImg_GaussianNoise, dstImg_blur_GaussianNoise, Size(5, 5));
    imshow("高斯噪声均值滤波后图像", dstImg_blur_GaussianNoise);//显示原图像
    */

    /*
    //中值滤波
    dstImg_median_SaltNoise= Mat(Rowsnum, Colsnum, CV_8UC3);
    median_filter(dstImg_SaltNoise,dstImg_median_SaltNoise, 3);
    imshow("椒盐噪声中值滤波后图像", dstImg_median_SaltNoise);
    //
    dstImg_median_GaussianNoise = Mat(Rowsnum, Colsnum, CV_8UC3);
    median_filter(dstImg_GaussianNoise, dstImg_median_GaussianNoise, 3);
    imshow("高斯噪声中值滤波后图像", dstImg_median_GaussianNoise);
    */

    //图像锐化
   // sharpen_FirstOrder(Img, dstImg_sharpen_FirstOrder);
    //imshow("一阶微分算子锐化后图像", dstImg_sharpen_FirstOrder);//显示一阶锐化后图像
    Sharpen_laplace(Img, dstImg_sharpen_SecondOrder);
    imshow("二阶拉普拉斯算子锐化后图像", dstImg_sharpen_SecondOrder);

    waitKey(0);
    return 0;
}

//均值滤波
void myblur(const Mat& src, Mat& dst, Size ksize)
{
    //若模板不是奇数模板，则报错退出
    if (ksize.width % 2 == 0 || ksize.height % 2 == 0)
    {
        cout << "please input odd ksize!" << endl;
        exit(-1);
    }
    //根据ksize大小扩充模板边界
    int awidth = (ksize.width - 1) / 2;
    int aheight = (ksize.height - 1) / 2;
    Mat asrc;
    copyMakeBorder(src, asrc, aheight, aheight, awidth, awidth, BORDER_DEFAULT);
    //根据图像通道数遍历图像求均值
    //通道数为1
    if (src.channels() == 1)
    {
        for (int i = aheight; i < src.rows + aheight; i++)
        {
            for (int j = awidth; j < src.cols + awidth; j++)
            {
                int sum = 0;
                int mean = 0;
                for (int k = i - aheight; k <= i + aheight; k++)
                {
                    for (int l = j - awidth; l <= j + awidth; l++)
                    {
                        sum += asrc.at<uchar>(k, l);
                    }
                }
                mean = sum / (ksize.width * ksize.height);
                dst.at<uchar>(i - aheight, j - awidth) = mean;
            }
        }
    }
    //通道数为3	

    if (src.channels() == 3)
    {
        for (int i = aheight; i < src.rows + aheight; i++)
        {
            for (int j = awidth; j < src.cols + awidth; j++)
            {
                int sum[3] = { 0 };
                int mean[3] = { 0 };
                for (int k = i - aheight; k <= i + aheight; k++)
                {
                    for (int l = j - awidth; l <= j + awidth; l++)
                    {
                        sum[0] += asrc.at<Vec3b>(k, l)[0];
                        sum[1] += asrc.at<Vec3b>(k, l)[1];
                        sum[2] += asrc.at<Vec3b>(k, l)[2];
                    }
                }
                for (int m = 0; m < 3; m++)
                {
                    mean[m] = sum[m] / (ksize.width * ksize.height);
                    dst.at<Vec3b>(i - aheight, j - awidth)[m] = mean[m];
                }
            }
        }
    }
}
//生成随机椒盐噪声
void addSaltNoise(const Mat& srcImage, Mat& dstImage, int n)
{
    dstImage = srcImage.clone();
    for (int k = 0; k < n; k++)
    {
        //随机取值行列
        int i = rand() % dstImage.rows;
        int j = rand() % dstImage.cols;
        //图像通道判定
        if (dstImage.channels() == 1)
        {
            dstImage.at<uchar>(i, j) = 255;		//盐噪声
        }
        else
        {
            dstImage.at<Vec3b>(i, j)[0] = 255;
            dstImage.at<Vec3b>(i, j)[1] = 255;
            dstImage.at<Vec3b>(i, j)[2] = 255;
        }
    }
    for (int k = 0; k < n; k++)
    {
        //随机取值行列
        int i = rand() % dstImage.rows;
        int j = rand() % dstImage.cols;
        //图像通道判定
        if (dstImage.channels() == 1)
        {
            dstImage.at<uchar>(i, j) = 0;		//椒噪声
        }
        else
        {
            dstImage.at<Vec3b>(i, j)[0] = 0;
            dstImage.at<Vec3b>(i, j)[1] = 0;
            dstImage.at<Vec3b>(i, j)[2] = 0;
        }
    }
}
//生成高斯噪声
double generateGaussianNoise(double mu, double sigma)
{
    //定义小值
    const double epsilon = numeric_limits<double>::min();
    static double z0, z1;
    static bool flag = false;
    flag = !flag;
    //flag为假构造高斯随机变量X
    if (!flag)
        return z1 * sigma + mu;
    double u1, u2;
    //构造随机变量
    do
    {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while (u1 <= epsilon);
    //flag为真构造高斯随机变量
    z0 = sqrt(-2.0 * log(u1)) * cos(2 * CV_PI * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(2 * CV_PI * u2);
    return z0 * sigma + mu;
}
//为图像加入高斯噪声
void addGaussianNoise(Mat& srcImag, Mat& dstImage, double mu, double sigma)
{
    dstImage = srcImag.clone();
    int channels = dstImage.channels();
    int rowsNumber = dstImage.rows;
    int colsNumber = dstImage.cols * channels;
    //推断图像的连续性
    if (dstImage.isContinuous())
    {
        colsNumber *= rowsNumber;
        rowsNumber = 1;
    }
    for (int i = 0; i < rowsNumber; i++)
    {
        for (int j = 0; j < colsNumber; j++)
        {
            //加入高斯噪声
            int val = dstImage.ptr<uchar>(i)[j] +
                generateGaussianNoise(mu, sigma) * 32;
            if (val < 0)
                val = 0;
            if (val > 255)
                val = 255;
            dstImage.ptr<uchar>(i)[j] = (uchar)val;
        }
    }
}
//中值滤波
void median_filter(const Mat& src, Mat& dst, int ksize)
{
    // 参数检查：内核大小必须为奇数，且大于等于 3
    if (ksize % 2 == 0 || ksize < 3) {
        throw std::invalid_argument("Kernel size must be odd and greater than or equal to 3.");
    }
    // 克隆源图像以保留其内容，并获取通道数和中心位置
    dst = src.clone();
    int channels = src.channels(); // 获取图像通道数
    int pos = (ksize - 1) / 2; // 定义内核中心位置的偏移量
    // 创建一个用于存储像素值的动态数组
    int* windows = new int[ksize * ksize];
    // 分别处理源图像的每个通道
    for (int c = 0; c < channels; c++) {
        // 循环遍历图像中的每个像素（除了边缘）
        for (int m = pos; m < src.rows - pos; m++) {
            for (int n = pos; n < src.cols - pos; n++) {
                int winpos = 0;
                // 循环获取内核中的像素值
                for (int i = -pos; i <= pos; i++) {
                    for (int j = -pos; j <= pos; j++) {
                        windows[winpos++] = src.at<Vec3b>(m + i, n + j)[c];
                    }
                }
                // 使用vector计算中值
                std::vector<int> window_vec(windows, windows + ksize * ksize);
                std::sort(window_vec.begin(), window_vec.end());
                int mid = window_vec[(ksize * ksize) / 2];
                dst.at<Vec3b>(m, n)[c] = mid; // 将中值设为输出图像中的像素值
            }
        }
        // 处理输出图像边缘：把内核沿边缘外的像素用内核内的像素填充
        for (int i = 0; i < pos; i++) {
            for (int j = pos; j < src.cols - pos; j++) {
                dst.at<Vec3b>(i, j)[c] = dst.at<Vec3b>(pos, j)[c]; // 上边缘
                dst.at<Vec3b>(src.rows - 1 - i, j)[c] = dst.at<Vec3b>(src.rows - 1 - pos, j)[c]; // 下边缘
            }
        }
        for (int j = 0; j < pos; j++) {
            for (int i = 0; i < src.rows; i++) {
                dst.at<Vec3b>(i, j)[c] = dst.at<Vec3b>(i, pos)[c]; // 左边缘
                dst.at<Vec3b>(i, src.cols - 1 - j)[c] = dst.at<Vec3b>(i, src.cols - 1 - pos)[c]; // 右边缘
            }
        }
    }
    // 释放窗口数组的内存
    delete[] windows;
    windows = nullptr;
}
//一阶微分算子锐化
void sharpen_FirstOrder(const Mat& src, Mat& dst)
{
    // 克隆源图像作为输出图像，并获取图像的行数、列数和通道数
    dst = src.clone();
    int rows = src.rows;
    int cols = src.cols;
    int channels = src.channels();

    // 定义一阶微分算子内核
    float kernel_data[3][3] = { {-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1} };
    Mat kernel(3, 3, CV_32F, kernel_data);

    // 对每个通道分别处理
    for (int c = 0; c < channels; c++) {
        // 克隆源图像的通道和输出图像的通道，并将其转换为灰度图像
        Mat src_ch = src.clone();
        Mat dst_ch = dst.clone();
        if (channels > 1) {
            cvtColor(src, src_ch, COLOR_BGR2GRAY);
            cvtColor(dst, dst_ch, COLOR_BGR2GRAY);
        }
        // 循环遍历图像中的每个像素
        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
                float pixel = 0.0;
                // 对于像素周围的 3x3 区域，应用一阶微分算子内核进行卷积计算
                for (int k = -1; k <= 1; k++) {
                    for (int l = -1; l <= 1; l++) {
                        pixel += src_ch.at<uchar>(i + k, j + l) * kernel.at<float>(k + 1, l + 1);
                    }
                }
                // 将卷积结果作为输出图像中的像素值
                dst_ch.at<uchar>(i, j) = saturate_cast<uchar>(pixel);
            }
        }
        // 如果原始图像是彩色图像，则将输出图像由灰度转换为彩色
        if (channels > 1) {
            cvtColor(dst_ch, dst, COLOR_GRAY2BGR);
        }
    }
}
//二阶拉普拉斯算子锐化
void Sharpen_laplace(Mat& inputImage,Mat& outputImage) {
    // 获取图像宽度和高度
    int width = inputImage.cols;
    int height = inputImage.rows;

    outputImage.create(inputImage.size(), inputImage.type());
    outputImage.setTo(Scalar(0, 0, 0));

    // 处理每个像素
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            // 计算当前像素的拉普拉斯值
            int laplaceValue = -4 * inputImage.at<Vec3b>(y, x)[0]
                + inputImage.at<cv::Vec3b>(y - 1, x)[0]
                + inputImage.at<cv::Vec3b>(y + 1, x)[0]
                + inputImage.at<cv::Vec3b>(y, x - 1)[0]
                + inputImage.at<cv::Vec3b>(y, x + 1)[0];

            // 将结果存储到输出图像中
            cv::Vec3b outputPixel(std::clamp(inputImage.at<Vec3b>(y, x)[0] + laplaceValue, 0, 255),
                std::clamp(inputImage.at<Vec3b>(y, x)[1] + laplaceValue, 0, 255),
                std::clamp(inputImage.at<Vec3b>(y, x)[2] + laplaceValue, 0, 255));

            outputImage.at<Vec3b>(y, x) = outputPixel;
        }
        }
        }