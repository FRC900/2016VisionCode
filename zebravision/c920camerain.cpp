#include <iostream>
#include <string>
#include "c920camerain.hpp"
using namespace std;
#ifdef __linux__
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

// Prototypes for various callbacks used by scrollbars
void brightnessCallback(int value, void *data);
void contrastCallback(int value, void *data);
void saturationCallback(int value, void *data);
void sharpnessCallback(int value, void *data);
void gainCallback(int value, void *data);
void autoExposureCallback(int value, void *data);
void backlightCompensationCallback(int value, void *data);
void whiteBalanceTemperatureCallback(int value, void *data);
void focusCallback(int value, void *data);

// Constructor
C920CameraIn::C920CameraIn(int _stream, bool gui, ZvSettings *settings) :
	MediaIn(settings), camera_(_stream >= 0 ? _stream : 0)
{
	if (!camera_.IsOpen())
		cerr << "Could not open C920 camera" << endl;
	else if (!initCamera(gui))
	{
		camera_.Close();
		cerr << "Camera is not a C920" << endl;
	}
}

bool
C920CameraIn::loadSettings()
{
	if (_settings) {
		_settings->getInt(getClassName(), "brightness",              brightness_);
		_settings->getInt(getClassName(), "contrast",                contrast_);
		_settings->getInt(getClassName(), "saturation",              saturation_);
		_settings->getInt(getClassName(), "sharpness",               sharpness_);
		_settings->getInt(getClassName(), "gain",                    gain_);
		_settings->getInt(getClassName(), "focus",                   focus_);
		_settings->getInt(getClassName(), "autoExposure",            autoExposure_);
		_settings->getInt(getClassName(), "backlightCompensation",   backlightCompensation_);
		_settings->getInt(getClassName(), "whiteBalanceTemperature", whiteBalanceTemperature_);
		return true;
	}
	return false;
}

bool
C920CameraIn::saveSettings()
{
	if (_settings) {
		_settings->setInt(getClassName(), "brightness",              brightness_);
		_settings->setInt(getClassName(), "contrast",                contrast_);
		_settings->setInt(getClassName(), "saturation",              saturation_);
		_settings->setInt(getClassName(), "sharpness",               sharpness_);
		_settings->setInt(getClassName(), "gain",                    gain_);
		_settings->setInt(getClassName(), "focus",                   focus_);
		_settings->setInt(getClassName(), "autoExposure",            autoExposure_);
		_settings->setInt(getClassName(), "backlightCompensation",   backlightCompensation_);
		_settings->setInt(getClassName(), "whiteBalanceTemperature", whiteBalanceTemperature_);
		_settings->save();
		return true;
	}
	return false;
}

bool C920CameraIn::initCamera(bool gui)
{
	brightness_ = 128;
	contrast_   = 128;
	saturation_ = 128;
	sharpness_  = 128;
	gain_       = 1;
	backlightCompensation_   = 0;
	whiteBalanceTemperature_ = 0;

  if (!loadSettings())
		cerr << "Failed to load C920 settings from XML file" << endl;

	// TODO - do we want to set these or go
	// with the values set above?
	//captureSize_ = v4l2::CAPTURE_SIZE_640x480;
  	captureSize_ = v4l2::CAPTURE_SIZE_1280x720;
	if (!camera_.ChangeCaptureSize(captureSize_))
	{
		return false;
	}
	if (!camera_.ChangeCaptureFPS(v4l2::CAPTURE_FPS_30))
	{
		return false;
	}
#if 0
	if (!camera_.GetBrightness(brightness_))
	{
		return false;
	}
	if (!camera_.GetContrast(contrast_))
	{
		return false;
	}
	if (!camera_.GetSaturation(saturation_))
	{
		return false;
	}
	if (!camera_.GetSharpness(sharpness_))
	{
		return false;
	}
	if (!camera_.GetGain(gain_))
	{
		return false;
	}
	if (!camera_.GetBacklightCompensation(backlightCompensation_))
	{
		return false;
	}
	if (!camera_.GetWhiteBalanceTemperature(whiteBalanceTemperature_))
	{
		return false;
	}
	++whiteBalanceTemperature_;
#endif

	// force focus to farthest distance, non-auto
	focusCallback(1, this);
	autoExposureCallback(3, this);

	if (gui)
	{
		cv::namedWindow("Adjustments", CV_WINDOW_NORMAL);
		cv::createTrackbar("Brightness", "Adjustments", &brightness_, 255, brightnessCallback, this);
		cv::createTrackbar("Contrast", "Adjustments", &contrast_, 255, contrastCallback, this);
		cv::createTrackbar("Saturation", "Adjustments", &saturation_, 255, saturationCallback, this);
		cv::createTrackbar("Sharpness", "Adjustments", &sharpness_, 255, sharpnessCallback, this);
		cv::createTrackbar("Gain", "Adjustments", &gain_, 255, gainCallback, this);
		cv::createTrackbar("Auto Exposure", "Adjustments", &autoExposure_, 3, autoExposureCallback, this);
		cv::createTrackbar("Backlight Compensation", "Adjustments", &backlightCompensation_, 1, backlightCompensationCallback, this);
		// Off by one to account for -1 being auto.
		cv::createTrackbar("White Balance Temperature", "Adjustments", &whiteBalanceTemperature_, 6501, whiteBalanceTemperatureCallback, this);
		cv::createTrackbar("Focus", "Adjustments", &focus_, 256, focusCallback, this);
	}

	frameNumber_ = 0;
	return true;
}

bool C920CameraIn::isOpened(void) const
{
	return camera_.IsOpen();
}

bool C920CameraIn::update(void)
{
	if (!camera_.IsOpen() ||
	    !camera_.GrabFrame() ||
	    !camera_.RetrieveMat(localFrame_))
		return false;
	boost::lock_guard<boost::mutex> guard(_mtx);
	localFrame_.copyTo(_frame);
	while (_frame.rows > 700)
		pyrDown(_frame, _frame);
	frameNumber_ += 1;
	return true;
}

bool C920CameraIn::getFrame(cv::Mat &frame, cv::Mat &depth, bool pause)
{
	(void)pause;
	if (!camera_.IsOpen())
		return false;
	depth = Mat();
	boost::lock_guard<boost::mutex> guard(_mtx);
	if( _frame.empty() )
		return false;
	_frame.copyTo(frame);
	lockedFrameNumber_ = frameNumber_;
	return true;
}

int C920CameraIn::width(void) const
{
	unsigned int width;
	unsigned int height;

	v4l2::GetCaptureSize(captureSize_, width, height);

	// getNextFrame sizes down large images
	// adjust width and height to match that
	while (height > 700)
	{
		width /= 2;
		height /= 2;
	}

	return width;
}

int C920CameraIn::height(void) const
{
	unsigned int width;
	unsigned int height;

	v4l2::GetCaptureSize(captureSize_, width, height);

	// getNextFrame sizes down large images
	// adjust width and height to match that
	while (height > 700)
	{
		width /= 2;
		height /= 2;
	}

	return height;
}

int C920CameraIn::frameNumber(void) const
{
	return lockedFrameNumber_;
}

CameraParams C920CameraIn::getCameraParams(bool left) const
{
	(void)left;
	unsigned int width;
	unsigned int height;

	v4l2::GetCaptureSize(captureSize_, width, height);
	CameraParams cp;
	if (height == 480)
		cp.fov = Point2f(69.0 * M_PI / 180., 69.0 * 480 / 640. * M_PI / 180.); // need VFOV, other resolutions
	else
		//cp.fov = Point2f(77. * M_PI / 180., 77. * 720. / 1280. * M_PI / 180.);
		cp.fov = Point2f(70.42 * M_PI / 180., 43.3 * M_PI / 180.);
	return cp;
}


void brightnessCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->brightness_ = value;
	camPtr->camera_.SetBrightness(value);
	camPtr->saveSettings();
}

void contrastCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->contrast_ = value;
	camPtr->camera_.SetContrast(value);
	camPtr->saveSettings();
}

void saturationCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->saturation_ = value;
	camPtr->camera_.SetSaturation(value);
	camPtr->saveSettings();
}

void sharpnessCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->sharpness_ = value;
	camPtr->camera_.SetSharpness(value);
	camPtr->saveSettings();
}

void gainCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->gain_ = value;
	camPtr->camera_.SetGain(value);
	camPtr->saveSettings();
}

void backlightCompensationCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->backlightCompensation_ = value;
	camPtr->camera_.SetBacklightCompensation(value);
	camPtr->saveSettings();
}

void autoExposureCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->autoExposure_ = value;
	camPtr->camera_.SetAutoExposure(value);
	camPtr->saveSettings();
}

void whiteBalanceTemperatureCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->whiteBalanceTemperature_ = value;
	// Off by one to allow -1=auto
	camPtr->camera_.SetWhiteBalanceTemperature(value - 1);
	camPtr->saveSettings();
}

void focusCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->focus_ = value;
	// Off by one to allow -1=auto
	camPtr->camera_.SetFocus(value - 1);
	camPtr->saveSettings();
}

#else

C920CameraIn::C920CameraIn(int _stream, bool gui)
{
	(void)_stream;
	(void)gui;
	std::cerr << "C920 support not enabled" << std::endl;
}

#endif
