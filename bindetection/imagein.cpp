#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "imagein.hpp"

using namespace cv;

ImageIn::ImageIn(const char *path) :
	frame_(imread(path))
{
	if (frame_.empty())
		std::cerr << "Could not open image file " << path << std::endl;
	while (frame_.rows > 800)
		pyrDown(frame_, frame_);
}

bool ImageIn::getNextFrame(Mat &frame, bool pause)
{
	if (frame_.empty())
		return false;
	frame = frame_.clone();
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

int ImageIn::frameCounter(void) const
{
	return 1;
}

