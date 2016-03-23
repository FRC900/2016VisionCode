#ifndef ZEDCAMERAIN_HPP__
#define ZEDCAMERAIN_HPP__

//opencv include
#include <opencv2/core/core.hpp>
#include "mediain.hpp"

#ifdef ZED_SUPPORT
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
//zed include
#include <zed/Mat.hpp>
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>
#endif

class ZedIn : public MediaIn
{
	public:
		ZedIn(const char *inFileName = NULL, bool gui = false);
		~ZedIn();
		bool update();
		bool getFrame(cv::Mat &frame, cv::Mat &depth);
		int semValue() { return semValue_; }

		int    width(void) const;
		int    height(void) const;

#ifdef ZED_SUPPORT
		// How many frames?
		int    frameCount(void) const;

		// Get and set current frame number
		int    frameNumber(void) const;
		void   frameNumber(int frameNumber);

		CameraParams getCameraParams(bool left) const;
#endif

	private:
#ifdef ZED_SUPPORT
		void deleteInputPointers(void);
		bool openSerializeInput(const char *filename);
		bool update(bool left);

		sl::zed::Camera* zed_;
		sl::zed::Mat slDepth_;
		sl::zed::Mat slFrame_;
		cv::Mat localFrame_;
		cv::Mat localDepth_;
		cv::Mat depthMat_;
		int width_;
		int height_;
		int frameNumber_;
		int lockedFrameNumber_;
		int semValue_;

		int brightness_;
		int contrast_;
		int hue_;
		int saturation_;
		int gain_;
		int whiteBalance_;

		// Hack up a way to save zed data - serialize both
		// BGR frame and depth frame
		std::ifstream *serializeIn_;
		boost::iostreams::filtering_streambuf<boost::iostreams::input> *filtSBIn_;
		boost::archive::binary_iarchive *archiveIn_;

		// Mark these as friends so they can access private class data
		friend void zedBrightnessCallback(int value, void *data);
		friend void zedContrastCallback(int value, void *data);
		friend void zedHueCallback(int value, void *data);
		friend void zedSaturationCallback(int value, void *data);
		friend void zedGainCallback(int value, void *data);
		friend void zedWhiteBalanceCallback(int value, void *data);
#endif
};
#endif
