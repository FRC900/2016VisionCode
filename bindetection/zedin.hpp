#ifndef ZEDCAMERAIN_HPP__
#define ZEDCAMERAIN_HPP__

//opencv include
#include <opencv2/highgui/highgui.hpp>
#include "mediain.hpp"

#ifdef ZED_SUPPORT
//zed include
#include <zed/Mat.hpp>
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>
#endif

class ZedIn : public MediaIn
{
	public:
		ZedIn();
		bool getNextFrame(cv::Mat &frame, bool pause = false);

		int width(void);
		int height(void);
#ifdef ZED_SUPPORT
		double getDepth(int x, int y);
#endif

	private:
#ifdef ZED_SUPPORT
		bool getNextFrame(cv::Mat &frame, bool left, bool pause);
		sl::zed::Camera* zed;
		sl::zed::Mat imageGPU;
		sl::zed::Mat depthGPU;
		cv::Mat depthCPU;
		sl::zed::Mat depthMat;
		int width_;
		int height_;
#endif
};
#endif

