#include <iostream>
#include "c920camerain.hpp"
using namespace std;
#ifdef __linux__
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

C920CameraIn::C920CameraIn(int _stream, bool gui) :
	_camera(_stream >= 0 ? _stream : 0)
{
	if (!_camera.IsOpen())
		cerr << "Could not open C920 camera" << endl;
	else if (!initCamera(_stream, gui))
	{
		_camera.Close();
		cerr << "Camera is not a C920" << endl;
	}
}

void brightnessCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->_brightness = value;
	camPtr->_camera.SetBrightness(value);
}

void contrastCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->_contrast = value;
	camPtr->_camera.SetContrast(value);
}

void saturationCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->_saturation= value;
	camPtr->_camera.SetSaturation(value);
}

void sharpnessCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->_sharpness = value;
	camPtr->_camera.SetSharpness(value);
}

void gainCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->_gain = value;
	camPtr->_camera.SetGain(value);
}

void backlightCompensationCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->_backlightCompensation = value;
	camPtr->_camera.SetBacklightCompensation(value);
}

void autoExposureCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->_autoExposure = value;
	camPtr->_camera.SetAutoExposure(value);
}

void whiteBalanceTemperatureCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->_whiteBalanceTemperature = value;
	// Off by one to allow -1=auto
	camPtr->_camera.SetWhiteBalanceTemperature(value - 1);
}

void focusCallback(int value, void *data)
{
	C920CameraIn *camPtr = (C920CameraIn *)data;
	camPtr->_focus = value;
	// Off by one to allow -1=auto
	camPtr->_camera.SetFocus(value - 1);
}

bool C920CameraIn::initCamera(int _stream, bool gui)
{
	_brightness = 128;
	_contrast   = 128;
	_saturation = 128;
	_sharpness  = 128;
	_gain       = 1;
	_backlightCompensation   = 0;
	_whiteBalanceTemperature = 0;

	// TODO - do we want to set these or go
	// with the values set above?
	_captureSize = v4l2::CAPTURE_SIZE_640x480;
	if (!_camera.ChangeCaptureSize(_captureSize))
	{
		return false;
	};
	_camera.ChangeCaptureFPS(v4l2::CAPTURE_FPS_30);
	{
		return false;
	};
	_camera.GetBrightness(_brightness);
	{
		return false;
	};
	_camera.GetContrast(_contrast);
	{
		return false;
	};
	_camera.GetSaturation(_saturation);
	{
		return false;
	};
	_camera.GetSharpness(_sharpness);
	{
		return false;
	};
	_camera.GetGain(_gain);
	{
		return false;
	};
	_camera.GetBacklightCompensation(_backlightCompensation);
	{
		return false;
	};
	_camera.GetWhiteBalanceTemperature(_whiteBalanceTemperature);
	{
		return false;
	};
	++_whiteBalanceTemperature;

	// force focus to farthest distance, non-auto
	focusCallback(1, this); 
	autoExposureCallback(1, this);

	if (gui)
	{
		cv::namedWindow("Adjustments", CV_WINDOW_NORMAL);
		cv::createTrackbar("Brightness", "Adjustments", &_brightness, 255, brightnessCallback, this);
		cv::createTrackbar("Contrast", "Adjustments", &_contrast, 255, contrastCallback, this);
		cv::createTrackbar("Saturation", "Adjustments", &_saturation, 255, saturationCallback, this);
		cv::createTrackbar("Sharpness", "Adjustments", &_sharpness, 255, sharpnessCallback, this);
		cv::createTrackbar("Gain", "Adjustments", &_gain, 255, gainCallback, this);
		cv::createTrackbar("Auto Exposure", "Adjustments", &_autoExposure, 3, autoExposureCallback, this);
		cv::createTrackbar("Backlight Compensation", "Adjustments", &_backlightCompensation, 1, backlightCompensationCallback, this);
		// Off by one to account for -1 being auto.
		cv::createTrackbar("White Balance Temperature", "Adjustments", &_whiteBalanceTemperature, 6501, whiteBalanceTemperatureCallback, this);
		cv::createTrackbar("Focus", "Adjustments", &_focus, 256, focusCallback, this);
	}

	_frameCounter = 0;
	return true;
}

bool C920CameraIn::getNextFrame(Mat &frame, bool pause)
{
	if (!_camera.IsOpen())
		return false;

	if (!pause)
	{
		if (_camera.GrabFrame())
			_camera.RetrieveMat(_frame);
		if( _frame.empty() )
			return false;
		if (_frame.rows > 800)
			pyrDown(_frame, _frame);
		_frameCounter += 1;
	}

	frame = _frame.clone();
	return true;
}

int C920CameraIn::width(void)
{
	return v4l2::CAPTURE_SIZE_WIDTHS[_captureSize];
}

int C920CameraIn::height(void)
{
	return v4l2::CAPTURE_SIZE_HEIGHTS[_captureSize];
}

#else

C920CameraIn::C920CameraIn(int _stream, bool gui)
{
	std::cerr << "C920 support not enabled" << std::endl;
}

bool C920CameraIn::getNextFrame(Mat &frame, bool pause)
{
	return false;
}

int C920CameraIn::width(void)
{
	return 0;
}

int C920CameraIn::height(void)
{
	return 0;
}

#endif
