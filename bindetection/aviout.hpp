#pragma once
#include <opencv2/highgui/highgui.hpp>
#include "mediaout.hpp"

class AVIOut : public MediaOut
{
	public:
		AVIOut(const char *outFile, const cv::Size &size, int frameSkip = 0);
		~AVIOut();
		bool saveFrame(const cv::Mat &frame, const cv::Mat &depth);
	private :
		bool openNext(void);

		cv::Mat          frame_;
		cv::Size         size_;
		cv::VideoWriter *writer_;
		std::string      fileName_;
		//
		// Skip output frames if requested.  Skip is how many to 
		// skip before writing next output frame, FrameCounter is how
		// many total frames seen.
		// Counter is used to split the output into multiple shorter
		// outputs
		int frameSkip_;
		int frameCounter_;
		int fileCounter_;
		
};
