#include <iostream>
#include <opencv2/opencv.hpp>

#include "camerain.hpp"

using namespace cv;
using namespace std;

CameraIn::CameraIn(int stream, ZvSettings *settings) :
	MediaIn(settings),
	saveWidth_(1280),
	saveHeight_(720),
	fps_(30.),
	cap_(stream)
{
	if (cap_.isOpened())
	{
		if (!loadSettings()) {
			cerr << "Failed to load CameraIn settings" << endl;
		}

		width_ = saveWidth_;
		height_ = saveHeight_;
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
		settings_->getInt(getClassName(), "width", saveWidth_);
		settings_->getInt(getClassName(), "height", saveHeight_);
		return true;
	}
	return false;
}

bool CameraIn::saveSettings(void) const
{
	if (settings_) {
		settings_->setDouble(getClassName(), "fps", fps_);
		settings_->setInt(getClassName(), "width", saveWidth_);
		settings_->setInt(getClassName(), "height", saveHeight_);
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
	if (!cap_.isOpened()  ||
	    !cap_.grab() ||
	    !cap_.retrieve(localFrame_))
		return false;
	boost::lock_guard<boost::mutex> guard(mtx_);
	setTimeStamp();
	incFrameNumber();
	localFrame_.copyTo(frame_);
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

int CameraIn::width(void) const
{
   return width_;
}

int CameraIn::height(void) const
{
   return height_;
}

