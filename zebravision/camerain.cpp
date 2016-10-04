#include <iostream>
#include <opencv2/opencv.hpp>

#include "camerain.hpp"

using namespace cv;
using namespace std;

CameraIn::CameraIn(int stream, ZvSettings *settings) :
	MediaIn(settings),
	fps_(30.),
	cap_(stream)
{
	if (cap_.isOpened())
	{
		// Defaults, might be overridden by saved 
		// setting
		width_ = 1280;
		height_ = 720;

		if (!loadSettings()) {
			cerr << "Failed to load CameraIn settings" << endl;
		}

		cap_.set(CV_CAP_PROP_FPS, fps_);
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

CameraIn::~CameraIn()
{
	saveSettings();
}

bool CameraIn::loadSettings(void)
{
	if (settings_) {
		settings_->getDouble(getClassName(), "fps", fps_);
		settings_->getUnsignedInt(getClassName(), "width", width_);
		settings_->getUnsignedInt(getClassName(), "height", height_);
		return true;
	}
	return false;
}

bool CameraIn::saveSettings(void) const
{
	if (settings_) {
		settings_->setDouble(getClassName(), "fps", fps_);
		settings_->setInt(getClassName(), "width", 
				cap_.get(CV_CAP_PROP_FRAME_WIDTH));
		settings_->setInt(getClassName(), "height", 
				cap_.get(CV_CAP_PROP_FRAME_HEIGHT));
		settings_->save();
		return true;
	}
	return false;
}

bool CameraIn::isOpened() const
{
	return cap_.isOpened();
}

bool CameraIn::update(void)
{
	FPSmark();
	Mat localFrame;
	if (!cap_.isOpened()  ||
	    !cap_.grab() ||
	    !cap_.retrieve(localFrame))
		return false;
	boost::lock_guard<boost::mutex> guard(mtx_);
	setTimeStamp();
	incFrameNumber();
	localFrame.copyTo(frame_);
	while (frame_.rows > 700)
		pyrDown(frame_, frame_);
	return true;
}

bool CameraIn::getFrame(Mat &frame, Mat &depth, bool pause)
{
	if (!cap_.isOpened())
		return false;
	if (!pause)
	{
		boost::lock_guard<boost::mutex> guard(mtx_);
		if (frame_.empty())
			return false;
		lockTimeStamp();
		lockFrameNumber();
		frame_.copyTo(pausedFrame_);
	}
	pausedFrame_.copyTo(frame);
	depth = Mat();
	return true;
}

