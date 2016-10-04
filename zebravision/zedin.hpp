#pragma once

//opencv include
#include <opencv2/core/core.hpp>
#include "mediain.hpp"

#ifdef ZED_SUPPORT
//zed include
#include <zed/Mat.hpp>
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>
#endif

class ZvSettings;

class ZedIn : public MediaIn
{
	public:
		ZedIn(ZvSettings *settings = NULL);
		~ZedIn(void);

#ifdef ZED_SUPPORT
		CameraParams getCameraParams(bool left) const;

	protected:
		sl::zed::Camera* zed_;
		sl::zed::Mat slDepth_;
		sl::zed::Mat slFrame_;
		cv::Mat pausedFrame_;
		cv::Mat pausedDepth_;
		cv::Mat depthMat_;

		int width_;
		int height_;

	private:
#endif
};
