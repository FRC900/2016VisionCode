#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camerain.hpp"

using namespace cv;

CameraIn::CameraIn(int stream, bool gui) :
	frameNumber_(0),
	width_(1280),
    height_(720),
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
		while (height_ > 700)
		{
			width_ /= 2;
			height_ /= 2;
		}
	}
	else
		std::cerr << "Could not open camera" << std::endl;
}

bool CameraIn::update() 
{
	if (!cap_.isOpened())
		return false;
	if (!cap_.grab())
		return false;
	if (!cap_.retrieve(localFrame_))
		return false;
	boost::lock_guard<boost::mutex> guard(_mtx);
	localFrame_.copyTo(_frame);
	while (_frame.rows > 700)
		pyrDown(_frame, _frame);
	frameNumber_ += 1;
	return true;
}

bool CameraIn::getFrame(Mat &frame, Mat &depth, bool pause)
{
	if (!cap_.isOpened())
		return false;
	depth = Mat();
	boost::lock_guard<boost::mutex> guard(_mtx);
	if (_frame.empty())
		return false;
	_frame.copyTo(frame);
	lockedFrameNumber_ = frameNumber_;
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
   return lockedFrameNumber_;
}
