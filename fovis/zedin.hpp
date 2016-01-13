//standard include
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <math.h>
#include <algorithm>
#include <stdint.h>

//opencv include
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

//zed include
#include <zed/Mat.hpp>
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>

//cuda include
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "npp.h"
#include "device_functions.h"

//#include "videoin_c920.hpp"

class ZedIn
{
   public:
	ZedIn();
        bool getNextFrame(bool pause, cv::Mat &frame);
	bool getNextFrame(cv::Mat &frame,bool pause,bool left);
	double getDepthPoint(int x, int y);
	uchar* getDepthData(bool pause);
	bool getNormalDepth(bool pause, cv::Mat &frame);
	int width;
	int height;
   private:
	sl::zed::Mat depthMat;
	sl::zed::Camera* zed;
	sl::zed::Mat imageGPU;
	sl::zed::Mat depthGPU;
	cv::Mat depthCPU;
	cv::Mat imageCPU;
};

