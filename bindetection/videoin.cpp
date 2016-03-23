#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#include "videoin.hpp"

using namespace cv;

VideoIn::VideoIn(const char *inpath) :
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

//this increment_ variable basically locks the update code to the speed of the getFrame loop.
//This is to make sure that we run detection on every frame of the video
bool VideoIn::update() 
{
	boost::lock_guard<boost::mutex> guard(_mtx);
	increment_ = true;
	return true;
}

bool VideoIn::getFrame(Mat &frame, Mat &depth)
{
	if (!cap_.isOpened())
		return false;
	boost::lock_guard<boost::mutex> guard(_mtx);
	if(increment_) 
	{
		cap_ >> _frame;
		if (_frame.empty())
			return false;
		while (_frame.rows > 800)
			pyrDown(_frame, _frame);
		frameNumber_ += 1;
	}
	increment_ = false;
	depth = Mat();
	_frame.copyTo(frame);
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
