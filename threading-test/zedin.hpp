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
	ZedIn();
	ZedIn(char* svo_path); //constructors

	bool update(); //call to pull a new frame

	double getDepthPoint(int x, int y);
	const cv::Mat getDepth() { return cv_depth; } //various ways to get image data from the camera
	const cv::Mat getFrame() { return cv_frame; }
	const cv::Mat getNormalDepth() { return cv_normalDepth; }
	const cv::Mat getConfidence() { return cv_confidence; }

	sl::zed::CamParameters getCameraParams() { if(_left) { return stereoParams.LeftCam; } else { return stereoParams.RightCam; } }
	sl::zed::StereoParameters getStereoParams() { return stereoParams; } //these are used for getting parameters for calibration-sensitive libraries

	void left(bool& left) { _left = left; } //functions for modifying which camera is currently being pulled
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

