#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camerain.hpp"

using namespace cv;

CameraIn::CameraIn(char *outfile, int stream, bool gui) :
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

		// open the output video
		if(outfile != NULL) {
			writer_.open(outfile, CV_FOURCC('M','J','P','G'), 15, Size(640, 480), true);
			if(!writer_.isOpened())
				std::cerr << "Could not open output video " << outfile << std::endl;
		}
	}
	else
		std::cerr << "Could not open camera" << std::endl;
}

bool CameraIn::update() {
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

bool CameraIn::getFrame(Mat &frame)
{
	boost::lock_guard<boost::mutex> guard(_mtx);
	if (!cap_.isOpened())
		return false;
	if (_frame.empty())
		return false;
	frame = _frame.clone();
	lockedFrameNumber_ = frameNumber_;
	return true;
}

bool CameraIn::saveFrame(cv::Mat &frame) {
	boost::lock_guard<boost::mutex> guard(_mtx);
	writer_ << frame;
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
