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
	lockedTimeStamp_ = setTimeStamp();
	while (frame_.rows > 800)
		pyrDown(frame_, frame_);
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

int ImageIn::width(void) const
{
	return frame_.cols;
}

int ImageIn::height(void) const
{
	return frame_.rows;
}

// Images have only 1 "frame"
int ImageIn::frameCount(void) const
{
	return 1;
}

int ImageIn::frameNumber(void) const
{
	return 1;
}
