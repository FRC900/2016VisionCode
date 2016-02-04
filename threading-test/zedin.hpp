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

class ZedIn
{
   public:
	ZedIn(const char*=NULL); //constructors

	bool update(); //call to pull a new frame

	double getDepthPoint(int x, int y);
	cv::Mat getDepth()const { return cv_depth; } //various ways to get image data from the camera
	cv::Mat getFrame()const { return cv_frame; }
	cv::Mat getNormalDepth()const { return cv_normalDepth; }
	cv::Mat getConfidence()const { return cv_confidence; }
	void getCopy(cv::Mat &frame) { cv_frame.copyTo(frame); }

	sl::zed::CamParameters getCameraParams() { if(_left) { return stereoParams.LeftCam; } else { return stereoParams.RightCam; } }
	sl::zed::StereoParameters getStereoParams() { return stereoParams; } //these are used for getting parameters for calibration-sensitive libraries

	void left(bool left) { _left = left; } //functions for modifying which camera is currently being pulled
	bool left() const { return _left; }

	int width() const { return _width; } //easy function for getting witdth and height (also inclded in getCameraParams)
	int height() const { return _height; }

   private:

	bool _left;
	int _width;
	int _height;

	sl::zed::Camera* zed;

	cv::Mat cv_depth;
	cv::Mat cv_normalDepth;
	cv::Mat cv_frame;
	cv::Mat cv_confidence;

	sl::zed::Mat imageGPU;
	
	sl::zed::StereoParameters stereoParams;
};

