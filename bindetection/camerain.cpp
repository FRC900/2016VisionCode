#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camerain.hpp"

using namespace cv;

CameraIn::CameraIn(int stream, bool gui) : 
	cap_(stream),
	width_(800),
    height_(600),
	frameCounter_(0)
{
	if (cap_.isOpened())
	{
		cap_.set(CV_CAP_PROP_FPS, 30.0);
		cap_.set(CV_CAP_PROP_FRAME_WIDTH, width_);
		cap_.set(CV_CAP_PROP_FRAME_HEIGHT, height_);
	}
	else
		std::cerr << "Could not open camera" << std::endl;
}

bool CameraIn::getNextFrame(Mat &frame, bool pause)
{
	if (!cap_.isOpened())
		return false;
	if (!pause)
	{
		cap_ >> frame_;
		if (frame_.empty())
			return false;
		if (frame_.rows > 800)
			pyrDown(frame_, frame_);
		frameCounter_ += 1;
	}
	frame = frame_.clone();

	return true;
}

int CameraIn::width(void) const
{
   return width_;
}

int CameraIn::height(void) const
{
   return height_;
}

int CameraIn::frameCounter(void) const
{
   return frameCounter_;
}
