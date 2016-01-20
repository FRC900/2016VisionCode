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

bool C920CameraIn::initCamera(int _stream, bool gui)
{
	_brightness = 128;
	_contrast   = 128;
	_saturation = 128;
	_sharpness  = 128;
	_gain       = 1;
	_focus      = 1;
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
	// _camera.GetFocus(_focus);
	// ++_focus;
	if (gui)
	{
		cv::namedWindow("Adjustments", CV_WINDOW_NORMAL);
		cv::createTrackbar("Brightness", "Adjustments", &_brightness, 255, brightnessCallback, this);
		cv::createTrackbar("Contrast", "Adjustments", &_contrast, 255);
		cv::createTrackbar("Saturation", "Adjustments", &_saturation, 255);
		cv::createTrackbar("Sharpness", "Adjustments", &_sharpness, 255);
		cv::createTrackbar("Gain", "Adjustments", &_gain, 255);
		cv::createTrackbar("Backlight Compensation", "Adjustments", &_backlightCompensation, 1);
		// Off by one to account for -1 being auto.
		cv::createTrackbar("White Balance Temperature", "Adjustments", &_whiteBalanceTemperature, 6501);
		cv::createTrackbar("Focus", "Adjustments", &_focus, 256);
	}

	_frameCounter = 0;
}

bool C920CameraIn::getNextFrame(Mat &frame, bool pause)
{
	if (!_camera.IsOpen())
		return false;
	// Maybe move these to onChange method?
	_camera.SetContrast(_contrast);
	_camera.SetSaturation(_saturation);
	_camera.SetSharpness(_sharpness);
	_camera.SetGain(_gain);
	_camera.SetBacklightCompensation(_backlightCompensation);
	_camera.SetWhiteBalanceTemperature(_whiteBalanceTemperature - 1);
	_camera.SetFocus(_focus - 1);
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
