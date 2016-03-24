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
C920CameraIn::C920CameraIn(int _stream, bool gui) :
	camera_(_stream >= 0 ? _stream : 0)
{
	if (!camera_.IsOpen())
		cerr << "Could not open C920 camera" << endl;
	else if (!initCamera(gui))
	{
		camera_.Close();
		cerr << "Camera is not a C920" << endl;
	}
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

bool C920CameraIn::update() 
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
}

void contrastCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->contrast_ = value;
	camPtr->camera_.SetContrast(value);
}

void saturationCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->saturation_ = value;
	camPtr->camera_.SetSaturation(value);
}

void sharpnessCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->sharpness_ = value;
	camPtr->camera_.SetSharpness(value);
}

void gainCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->gain_ = value;
	camPtr->camera_.SetGain(value);
}

void backlightCompensationCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->backlightCompensation_ = value;
	camPtr->camera_.SetBacklightCompensation(value);
}

void autoExposureCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->autoExposure_ = value;
	camPtr->camera_.SetAutoExposure(value);
}

void whiteBalanceTemperatureCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->whiteBalanceTemperature_ = value;
	// Off by one to allow -1=auto
	camPtr->camera_.SetWhiteBalanceTemperature(value - 1);
}

void focusCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->focus_ = value;
	// Off by one to allow -1=auto
	camPtr->camera_.SetFocus(value - 1);
}

#else

C920CameraIn::C920CameraIn(int _stream, bool gui)
{
	(void)_stream;
	(void)gui;
	std::cerr << "C920 support not enabled" << std::endl;
}

#endif
