#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camerain.hpp"

using namespace cv;

CameraIn::CameraIn(int stream, bool gui) : 
	frameNumber_(0),
	width_(800),
    height_(600),
	cap_(stream)
{
	(void)gui;
	if (cap_.isOpened())
	{
		cap_.set(CV_CAP_PROP_FPS, 30.0);
		cap_.set(CV_CAP_PROP_FRAME_WIDTH, width_);
		cap_.set(CV_CAP_PROP_FRAME_HEIGHT, height_);
		// getNextFrame resizes large inputs,
		// make sure width and height match
		while (height_ > 800)
		{
			width_ /= 2;
			height_ /= 2;
		}
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
		while (frame_.rows > 800)
			pyrDown(frame_, frame_);
		frameNumber_ += 1;
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

int CameraIn::frameNumber(void) const
{
   return frameNumber_;
}
