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
	if (!zed_)
		return;

	CamParameters zedp;

	if (left)
		zedp = zed_->getParameters()->LeftCam;
	else
		zedp = zed_->getParameters()->RightCam;

	float hFovDegrees;
	if (height_ == 480) // can't work based on width, since 1/2 of 1280x720 is 640, as is full sized 640x480
		hFovDegrees = 51.3;
	else
		hFovDegrees = 105.; // hope all the HD & 2k res are the same
	float hFovRadians = hFovDegrees * M_PI / 180.0;

	// Convert from ZED-specific to generic
	// params data type
	params_.fov = Point2f(hFovRadians, hFovRadians * (float)height_ / (float)width_);
	params_.fx = zedp.fx;
	params_.fy = zedp.fy;
	params_.cx = zedp.cx;
	params_.cy = zedp.cy;
}

#else


ZedIn::ZedIn(ZvSettings *settings) :
	MediaIn(settings)
{
	cerr << "ZED support not compiled in" << endl;
}

ZedIn::~ZedIn()
{
}


#endif
