#ifndef ZEDCAMERAIN_HPP__
#define ZEDCAMERAIN_HPP__

//opencv include
#include <opencv2/core/core.hpp>
#include "mediain.hpp"

#ifdef ZED_SUPPORT
#include <boost/archive/binary_oarchive.hpp>
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
		ZedIn(const char *inFileName = NULL, const char *outFileName = NULL, bool gui = false);
		~ZedIn();
		bool update();
		bool getFrame(cv::Mat &frame);

		int    width(void) const;
		int    height(void) const;

#ifdef ZED_SUPPORT
		// How many frames?
		int    frameCount(void) const;

		// Get and set current frame number
		int    frameNumber(void) const;
		void   frameNumber(int frameNumber);

		CameraParams getCameraParams(bool left) const;
		bool  getDepthMat(cv::Mat &depthMat) const;
		float getDepth(int x, int y);
		bool  getNormDepthMat(cv::Mat &depthMat) const;
#endif

	private:
#ifdef ZED_SUPPORT
		void deletePointers(void);
		void deleteInputPointers(void);
		void deleteOutputPointers(void);
		bool openSerializeInput(const char *filename);
		bool openSerializeOutput(const char *filename);
		bool update(bool left);

		sl::zed::Camera* zed_;
		cv::Mat frameRGBA_;
		cv::Mat depthMat_;
		cv::Mat normDepthMat_;
		int width_;
		int height_;
		int frameNumber_;

		int brightness_;
		int contrast_;
		int hue_;
		int saturation_;
		int gain_;
		int whiteBalance_;

		std::string outFileName_;

		// Hack up a way to save zed data - serialize both
		// BGR frame and depth frame
		std::ifstream *serializeIn_;
		boost::iostreams::filtering_streambuf<boost::iostreams::input> *filtSBIn_;
		boost::archive::binary_iarchive *archiveIn_;
		std::ofstream *serializeOut_;
		boost::iostreams::filtering_streambuf<boost::iostreams::output> *filtSBOut_;
		boost::archive::binary_oarchive *archiveOut_;

		// Mark these as friends so they can access private class data
		friend void zedBrightnessCallback(int value, void *data);
		friend void zedContrastCallback(int value, void *data);
		friend void zedHueCallback(int value, void *data);
		friend void zedSaturationCallback(int value, void *data);
		friend void zedGainCallback(int value, void *data);
		friend void zedWhiteBalanceCallback(int value, void *data);
		int serializeFrameStart_;
		int serializeFrameSize_;
#endif
};
#endif
