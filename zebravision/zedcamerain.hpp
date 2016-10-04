#pragma once

//opencv include
#include <opencv2/core/core.hpp>
#include "zedin.hpp"

#ifdef ZED_SUPPORT
#endif

class ZvSettings;

class ZedCameraIn : public ZedIn
{
	public:
		ZedCameraIn(bool gui = false, ZvSettings *settings = NULL);
		~ZedCameraIn();

#ifdef ZED_SUPPORT
		bool isOpened(void) const;
		bool update(void);
		bool getFrame(cv::Mat &frame, cv::Mat &depth, bool pause = false);
#endif

	private:
#ifdef ZED_SUPPORT
		bool update(bool left);
		bool loadSettings(void);
		bool saveSettings(void) const;
		std::string getClassName() const { return "ZedCameraIn"; }

		sl::zed::Camera* zed_;
		sl::zed::Mat slDepth_;
		sl::zed::Mat slFrame_;
		cv::Mat pausedFrame_;
		cv::Mat pausedDepth_;
		cv::Mat depthMat_;

		int brightness_;
		int contrast_;
		int hue_;
		int saturation_;
		int gain_;

		// Mark these as friends so they can access private class data
		friend void zedBrightnessCallback(int value, void *data);
		friend void zedContrastCallback(int value, void *data);
		friend void zedHueCallback(int value, void *data);
		friend void zedSaturationCallback(int value, void *data);
		friend void zedGainCallback(int value, void *data);
#endif
};
