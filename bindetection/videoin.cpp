#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#include "videoin.hpp"

using namespace cv;

VideoIn(const char *inpath, const char *outpath) :
	cap_(inpath)
{
	isVideo = true;
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

		// open the output video
		if(outpath != NULL) {
			writer_.open(*outpath, CV_FOURCC('M','J','P','G'), 15, Size(frame.cols, frame.rows), true);
			if(!writer_.isOpened())
				std::cerr << "Could not open output video" << outpath << std::endl;
		}
	}
	else
		std::cerr << "Could not open input video "<< inpath << std::endl;


}

//this increment variable basically locks the update code to the speed of the getFrame loop.
//This is to make sure that we run detection on every frame of the video
bool VideoIn::update() {
	increment = true;
	return true;
}

bool VideoIn::getFrame(Mat &frame)
{
	if(increment) {
		boost::lock_guard<boost::mutex> guard(_mtx);
		if (!cap_.isOpened())
			return false;
			cap_ >> _frame;
			if (_frame.empty())
				return false;
			while (_frame.rows > 800)
				pyrDown(_frame, _frame);
		frameNumber_ += 1;
	}
	increment = false;
	frame = _frame.clone();
	return true;
}

bool saveFrame(const cv::Mat &frame) {
	if (outputVideo.isOpened()) {
		writer_ << frame;
		return true;
	} else {
		return false;
	}
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
