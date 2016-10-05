#include <iostream>
#include "zedin.hpp"
using namespace std;

#ifdef ZED_SUPPORT
#include <opencv2/imgproc/imgproc.hpp>
#include "ZvSettings.hpp"

using namespace cv;
using namespace sl::zed;

ZedIn::ZedIn(ZvSettings *settings) :
	MediaIn(settings),
	zed_(NULL)
{
}


ZedIn::~ZedIn()
{
	if (zed_)
		delete zed_;
}


CameraParams ZedIn::getCameraParams(void) const
{
	return params_;
}

// For whatever reason, getParameters calls grab()
// This causes problems with the update() thread
// Could mutex protect this, but the easier way
// is to just set it once in the constructor once
// a Zed object is opened and then from there on return
// the results 
void ZedIn::initCameraParams(bool left)
{
	CamParameters zedp;
	if (zed_)
	{
		if (left)
			zedp = zed_->getParameters()->LeftCam;
		else
			zedp = zed_->getParameters()->RightCam;
	}
	else
	{
		// Take a guess based on acutal values from one of our cameras
		if (height_ == 480)
		{
			zedp.fx = 705.768;
			zedp.fy = 705.768;
			zedp.cx = 326.848;
			zedp.cy = 240.039;
		}
		else if ((width_ == 1280) || (width_ == 640)) // 720P normal or pyrDown 1x
		{
			zedp.fx = 686.07;
			zedp.fy = 686.07;
			zedp.cx = 662.955 / (1280 / width_);;
			zedp.cy = 361.614 / (1280 / width_);;
		}
		else if ((width_ == 1920) || (width_ == 960)) // 1920 downscaled
		{
			zedp.fx = 1401.88;
			zedp.fy = 1401.88;
			zedp.cx = 977.193 / (1920 / width_);; // Is this correct - downsized
			zedp.cy = 540.036 / (1920 / width_);; // image needs downsized cx?
		}
		else if ((width_ == 2208) || (width_ == 1104)) // 2208 downscaled
		{
			zedp.fx = 1385.4;
			zedp.fy = 1385.4;
			zedp.cx = 1124.74 / (2208 / width_);;
			zedp.cy = 1124.74 / (2208 / width_);;
		}
		else
		{
			// This should never happen
			zedp.fx = 0;
			zedp.fy = 0;
			zedp.cx = 0;
			zedp.cy = 0;
		}
	}
	float hFovDegrees;
	if (height_ == 480) // can't work based on width, since 1/2 of 720P is 640, as is 640x480
		hFovDegrees = 51.3;
	else
		hFovDegrees = 105.; // hope all the HD & 2k res are the same
	float hFovRadians = hFovDegrees * M_PI / 180.0;

	params_.fov = Point2f(hFovRadians, hFovRadians * (float)height_ / (float)width_);
	cout << height_ << "X" << width_ << endl;
	params_.fx = zedp.fx;
	params_.fy = zedp.fy;
	params_.cx = zedp.cx;
	params_.cy = zedp.cy;
}

#else


ZedIn::ZedIn(bool gui, ZvSettings *settings) :
	MediaIn(settings)
{
	(void)gui;
	cerr << "Zed support not compiled in" << endl;
}

ZedIn::~ZedIn()
{
}


#endif
