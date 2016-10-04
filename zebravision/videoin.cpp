#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#include "videoin.hpp"
#include "ZvSettings.hpp"

using namespace cv;

VideoIn::VideoIn(const char *inpath, ZvSettings *settings) :
	MediaIn(settings),
	cap_(inpath),
	frameReady_(false)
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

bool VideoIn::update(void)
{
	FPSmark();

	// If the frame read from the last update()
	// call hasn't been used yet, loop here
	// until it has been. This will prevent
	// the code from reading multiple frames
	// in the time it takes to process one and
	// skipping some video in the process
	boost::mutex::scoped_lock guard(mtx_);
	while (frameReady_)
		condVar_.wait(guard);

	cap_ >> frame_;
	if (frame_.empty())
	{
		frameReady_ = true;
		condVar_.notify_all();
		return false;
	}
	setTimeStamp();
	incFrameNumber();
	while (frame_.rows > 700)
		pyrDown(frame_, frame_);

	// Let getFrame know that a frame is ready
	// to be read / processed
	frameReady_ = true;
	condVar_.notify_all();

	return true;
}

bool VideoIn::getFrame(Mat &frame, Mat &depth, bool pause)
{
	if (!cap_.isOpened())
		return false;

	// Wait until a valid frame is in frame_
	boost::mutex::scoped_lock guard(mtx_);
	while (!frameReady_)
		condVar_.wait(guard);
	if (frame_.empty())
		return false;

	depth = Mat();
	frame_.copyTo(frame);
	lockTimeStamp();
	lockFrameNumber();

	// If paused, don't request a new 
	// frame from update() yet
	if (!pause)
	{
		frameReady_ = false;
		condVar_.notify_all();
	}
	return true;
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
		frameReady_ = false;
	}
}
