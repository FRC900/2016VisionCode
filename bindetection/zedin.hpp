#ifndef ZEDCAMERAIN_HPP__
#define ZEDCAMERAIN_HPP__

//opencv include
#include <opencv2/core/core.hpp>
#include "camerain.hpp"

#ifdef ZED_SUPPORT
//zed include
#include <zed/Mat.hpp>
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>
#endif

class ZedIn : public CameraIn
{
	public:
		ZedIn(const char *filename = NULL);
		bool getNextFrame(cv::Mat &frame, bool pause = false);

#ifdef ZED_SUPPORT
		double getDepth(int x, int y);
#endif

	private:
#ifdef ZED_SUPPORT
		bool getNextFrame(cv::Mat &frame, bool left, bool pause);
		sl::zed::Camera* zed;
		sl::zed::Mat imageGPU;
		sl::zed::Mat depthGPU;
		sl::zed::Mat depthMat;
		cv::Mat frameRGBA_;
#endif
};
#endif

