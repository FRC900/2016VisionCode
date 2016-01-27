#ifndef __linux__
// The C920 specific code only works under Linux. For windows, 
// uss the default OpenCV VideoCapture code instead.  Users
// of other cameras should hack this up to use the same code
#include "videoin.cpp"
#else
#include "videoin_c920.hpp"
#include <iostream>

using namespace cv;
using namespace std;

VideoIn::VideoIn(const char *path)
{
   // A path can be a still imaged
   if (strstr(path, ".png") || strstr(path, ".jpg"))
   {
      _frame = imread(path);
      _video = false;
   }
   else // or a video image
   {
      _cap = VideoCapture(path);
      _video = true;
   }
   _frameCounter = 0;
   _c920 = false;
}

VideoIn::VideoIn(int _stream, bool gui) :
	_camera(_stream >= 0 ? _stream : 0)
{
	if (!_camera.IsOpen())
	{
		std::cerr << "Could not open C920 camera" << std::endl;
		return;
	}
	else if (!initCamera(_stream, gui))
	{
		_camera.Close();
		std::cerr << "Camera is not a C920" << std::endl;
		return;
	}
   _frameCounter = 0;
	_video = true;
	_c920  = true;
}

void brightnessCallback(int value, void *data)
{
	VideoIn *camPtr = (VideoIn *)data;
	camPtr->_brightness = value;
	camPtr->_camera.SetBrightness(value);
}

void contrastCallback(int value, void *data)
{
	VideoIn *camPtr = (VideoIn *)data;
	camPtr->_contrast = value;
	camPtr->_camera.SetContrast(value);
}

void saturationCallback(int value, void *data)
{
	VideoIn *camPtr = (VideoIn *)data;
	camPtr->_saturation= value;
	camPtr->_camera.SetSaturation(value);
}

void sharpnessCallback(int value, void *data)
{
	VideoIn *camPtr = (VideoIn *)data;
	camPtr->_sharpness = value;
	camPtr->_camera.SetSharpness(value);
}

void gainCallback(int value, void *data)
{
	VideoIn *camPtr = (VideoIn *)data;
	camPtr->_gain = value;
	camPtr->_camera.SetGain(value);
}

void backlightCompensationCallback(int value, void *data)
{
	VideoIn *camPtr = (VideoIn *)data;
	camPtr->_backlightCompensation = value;
	camPtr->_camera.SetBacklightCompensation(value);
}

void whiteBalanceTemperatureCallback(int value, void *data)
{
	VideoIn *camPtr = (VideoIn *)data;
	camPtr->_whiteBalanceTemperature = value;
	// Off by one to allow -1=auto
	camPtr->_camera.SetWhiteBalanceTemperature(value - 1);
}

void focusCallback(int value, void *data)
{
	VideoIn *camPtr = (VideoIn *)data;
	camPtr->_focus = value;
	// Off by one to allow -1=auto
	camPtr->_camera.SetFocus(value - 1);
}

bool VideoIn::initCamera(int _stream, bool gui)
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
	}
	if (!_camera.ChangeCaptureFPS(v4l2::CAPTURE_FPS_30))
	{
		return false;
	}
	if (!_camera.GetBrightness(_brightness))
	{
		return false;
	}
	if (!_camera.GetContrast(_contrast))
	{
		return false;
	}
	if (!_camera.GetSaturation(_saturation))
	{
		return false;
	}
	if (!_camera.GetSharpness(_sharpness))
	{
		return false;
	}
	if (!_camera.GetGain(_gain))
	{
		return false;
	}
	if (!_camera.GetBacklightCompensation(_backlightCompensation))
	{
		return false;
	}
	if (!_camera.GetWhiteBalanceTemperature(_whiteBalanceTemperature))
	{
		return false;
	}
	++_whiteBalanceTemperature;

	// force focus to farthest distance, non-auto
	focusCallback(1, this); 

	if (gui)
	{
		cv::namedWindow("Adjustments", CV_WINDOW_NORMAL);
		cv::createTrackbar("Brightness", "Adjustments", &_brightness, 255, brightnessCallback, this);
		cv::createTrackbar("Contrast", "Adjustments", &_contrast, 255, contrastCallback, this);
		cv::createTrackbar("Saturation", "Adjustments", &_saturation, 255, saturationCallback, this);
		cv::createTrackbar("Sharpness", "Adjustments", &_sharpness, 255, sharpnessCallback, this);
		cv::createTrackbar("Gain", "Adjustments", &_gain, 255, gainCallback, this);
		cv::createTrackbar("Backlight Compensation", "Adjustments", &_backlightCompensation, 1, backlightCompensationCallback, this);
		// Off by one to account for -1 being auto.
		cv::createTrackbar("White Balance Temperature", "Adjustments", &_whiteBalanceTemperature, 6501, whiteBalanceTemperatureCallback, this);
		cv::createTrackbar("Focus", "Adjustments", &_focus, 256, focusCallback, this);
	}

	_frameCounter = 0;
	return true;
}

bool VideoIn::getNextFrame(bool pause, Mat &frame)
{
	if (!pause && _video)
	{
		if (_c920)
		{
			if (_camera.GrabFrame())
				_camera.RetrieveMat(_frame);
		}
		else
			_cap >> _frame;
		if( _frame.empty() )
			return false;
		if (_frame.rows > 800)
			pyrDown(_frame, _frame);
		_frameCounter += 1;
	}
	frame = _frame.clone();

	return true;
}

int VideoIn::frameCounter(void)
{
   return _frameCounter;
}

void VideoIn::frameCounter(int frameCount)
{
   if (_video && !_c920)
      _cap.set(CV_CAP_PROP_POS_FRAMES, frameCount);
   _frameCounter = frameCount;
}

VideoCapture *VideoIn::VideoCap(void) 
{
   if (_video && !_c920)
      return &_cap;
   return NULL;
}

#endif
