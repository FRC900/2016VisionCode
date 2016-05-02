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
		frameNumber_ = 0;
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
	if(!pause)
	{
		cap_ >> _frame;
		if (_frame.empty())
			return false;
		lockedTimeStamp_ = setTimeStamp();
		while (_frame.rows > 800)
			pyrDown(_frame, _frame);
		frameNumber_ += 1;
	}
	depth = Mat();
	_frame.copyTo(frame);
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
	std::cout << "Frames = " << frames_ << std::endl;
	return frames_;
}

int VideoIn::frameNumber(void) const
{
	return frameNumber_;
}

void VideoIn::frameNumber(int frameNumber)
{
	if (frameNumber < frames_)
	{
		cap_.set(CV_CAP_PROP_POS_FRAMES, frameNumber);
		frameNumber_ = frameNumber;
	}
}
