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
		std::cerr << "Could not open input video "<< path << std::endl;
}

bool VideoIn::update() {
	boost::lock_guard<boost::mutex> guard(_mtx);
	if (!cap_.isOpened())
		return false;
		cap_ >> _frame;
		if (_frame.empty())
			return false;
		while (_frame.rows > 800)
			pyrDown(_frame, _frame);
		frameNumber_ += 1;
			return true;
}

bool VideoIn::getFrame(Mat &frame)
{
	boost::lock_guard<boost::mutex> guard(_mtx);
	if (!cap_.isOpened())
		return false;
		if (_frame.empty())
			return false;
	frame = _frame.clone();

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
