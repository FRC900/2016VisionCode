#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#include "videoin.hpp"

using namespace cv;

VideoIn::VideoIn(const char *path) :
	cap_(path)
{
	if (cap_.isOpened())
	{
		width_  = cap_.get(CV_CAP_PROP_FRAME_WIDTH);
		height_ = cap_.get(CV_CAP_PROP_FRAME_WIDTH);
		// getNextFrame scales down large inputs
		// make width and height match adjusted frame size
		while (height_ > 800)
		{
			width_ /= 2;
			height_ /= 2;
		}
		frames_ = cap_.get(CV_CAP_PROP_FRAME_COUNT);
		frameCounter_ = 0;
	}
	else
		std::cerr << "Could not open input video "<< path;
}

bool VideoIn::getNextFrame(Mat &frame, bool pause)
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
		frameCounter_ += 1;
	}
	frame = frame_.clone();

	return true;
}

int VideoIn::width() const
{
	return width_;
}

int VideoIn::height() const
{
	return height_;
}

int VideoIn::frameCount(void) const
{
	return frames_;
}

int VideoIn::frameCounter(void) const
{
	return frameCounter_;
}

void VideoIn::frameCounter(int frameCounter)
{
	if (frameCounter < frames_)
	{
		cap_.set(CV_CAP_PROP_POS_FRAMES, frameCounter);
		frameCounter_ = frameCounter;
	}
}
