#pragma once
#include <opencv2/highgui/highgui.hpp>
#include "mediaout.hpp"

// Class to write MJPEG AVI files using
// OpenCV's VideoWriter class
class AVIOut : public MediaOut
{
	public:
		AVIOut(const char *outFile, const cv::Size &size, int frameSkip = 1);
		~AVIOut();

	private :
		bool openNext(void);
		bool write(const cv::Mat &frame, const cv::Mat &depth);

		cv::Size         size_;
		cv::VideoWriter *writer_;
		std::string      fileName_;
};
