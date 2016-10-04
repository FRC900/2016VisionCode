#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "imagein.hpp"
#include "ZvSettings.hpp"

using namespace cv;

ImageIn::ImageIn(const char *inpath, ZvSettings *settings) :
	MediaIn(settings)
{
	imread(inpath).copyTo(frame_);
	if (frame_.empty())
	{
		std::cerr << "Could not open image file " << inpath << std::endl;
		return;
	}
	while (frame_.rows > 700)
		pyrDown(frame_, frame_);

	width_ = frame_.cols;
	height_ = frame_.rows;
}

bool ImageIn::isOpened(void) const
{
	return frame_.empty();
}

bool ImageIn::update(void)
{
	usleep(250000);
	return true;
}

bool ImageIn::getFrame(Mat &frame, Mat &depth, bool pause)
{
	(void)pause;
	if (frame_.empty())
		return false;
	frame = frame_.clone();
	depth = Mat();
	return true;
}

// Images have only 1 "frame"
int ImageIn::frameCount(void) const
{
	return 1;
}

