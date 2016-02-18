#ifndef ZEDCAMERAIN_HPP__
#define ZEDCAMERAIN_HPP__

//opencv include
#include <opencv2/core/core.hpp>
#include "mediain.hpp"

#ifdef ZED_SUPPORT
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <fstream>
//zed include
#include <zed/Mat.hpp>
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>
#endif

class ZedIn : public MediaIn
{
	public:
		ZedIn(const char *inFileName = NULL, const char *outFileName = NULL);
		~ZedIn();
		bool getNextFrame(cv::Mat &frame, bool pause = false);

		int    width(void) const;
		int    height(void) const;

#ifdef ZED_SUPPORT
		// How many frames?
		int    frameCount(void) const; 

		// Get and set current frame number
		int    frameCounter(void) const;
		void   frameCounter(int framecount);

		void   deletePointers(void);
		bool   getDepthMat(cv::Mat &depthMat);
		double getDepth(int x, int y);
#endif

	private:
#ifdef ZED_SUPPORT
		bool getNextFrame(cv::Mat &frame, bool left, bool pause);

		sl::zed::Camera* zed_;
		cv::Mat frameRGBA_;
		cv::Mat frame_;
		cv::Mat depthMat_;
		int width_;
		int height_;
		int frameCounter_;

		// Hack up a way to save zed data - serialize both 
		// BGR frame and depth frame
		std::ifstream *serializeIn_;
		boost::iostreams::filtering_streambuf<boost::iostreams::input> filtSBIn_;
		boost::archive::binary_iarchive *archiveIn_;
		std::ofstream *serializeOut_;
		boost::iostreams::filtering_streambuf<boost::iostreams::output> filtSBOut_;
		boost::archive::binary_oarchive *archiveOut_;
		int serializeFrameSize_;
#endif
};
#endif

