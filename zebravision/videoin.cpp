#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#include "videoin.hpp"
#include "ZvSettings.hpp"

using namespace cv;

VideoIn::VideoIn(const char *inpath, ZvSettings *settings) :
	MediaIn(settings),
	cap_(inpath)
{
	if (cap_.isOpened())
	{
		width_  = cap_.get(CV_CAP_PROP_FRAME_WIDTH);
		height_ = cap_.get(CV_CAP_PROP_FRAME_HEIGHT);
		// getNextFrame scales down large inputs
		// make width and height match adjusted frame size
		while (height_ > 800)
		{
			width_ /= 2;
			height_ /= 2;
		}
		frames_ = cap_.get(CV_CAP_PROP_FRAME_COUNT);
	}
	else
		std::cerr << "Could not open input video "<< inpath << std::endl;
}

bool VideoIn::isOpened(void) const
{
	return cap_.isOpened();
}

// Do nothing - all of the work is acutally in getFrame
bool VideoIn::update(void)
{
	usleep(150000);
	return true;
}

bool VideoIn::getFrame(Mat &frame, Mat &depth, bool pause)
{
	if (!cap_.isOpened())
		return false;
	if (!pause)
	{
		cap_ >> frame_;
		if (frame_.empty())
			return false;
		setTimeStamp();
		incFrameNumber();
		while (frame_.rows > 800)
			pyrDown(frame_, frame_);
	}
	depth = Mat();
	frame_.copyTo(frame);
	lockTimeStamp();
	lockFrameNumber();
	return true;
}

int VideoIn::width(void) const
{
	return width_;
}

int VideoIn::height(void) const
{
	return height_;
}

int VideoIn::frameCount(void) const
{
	return frames_;
}

void VideoIn::frameNumber(int frameNumber)
{
	if (frameNumber < frames_)
	{
		cap_.set(CV_CAP_PROP_POS_FRAMES, frameNumber);
		setFrameNumber(frameNumber);
	}
}
