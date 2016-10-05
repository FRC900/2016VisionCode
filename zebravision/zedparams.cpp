#include <iostream>
#include "zedparams.hpp"

using namespace std;

ZedParams::ZedParams(void)
{
}

#ifdef ZED_SUPPORT

using namespace sl::zed;

CameraParams ZedParams::get(void) const
{
	return params_;
}

// For whatever reason, getParameters calls grab()
// This causes problems with the update() thread
// Could mutex protect this, but the easier way
// is to just set it once in the constructor once
// a Zed object is opened and then from there on return
// the results 
void ZedParams::init(sl::zed::Camera *zed, bool left)
{
	if (!zed)
		return;

	CamParameters zedp;

	if (left)
		zedp = zed->getParameters()->LeftCam;
	else
		zedp = zed->getParameters()->RightCam;

	unsigned height = zed->getImageSize().height;
	unsigned width  = zed->getImageSize().width;
	// Pick FOV based on 4:3 or 16:9 aspect ratio
	float hFovDegrees;
	if (height == 480) // 640x480. TODO :: widescreen VGA?
		hFovDegrees = 51.3;
	else
		hFovDegrees = 105.; // hope all the HD & 2k res are the same
	float hFovRadians = hFovDegrees * M_PI / 180.0;

	// Convert from ZED-specific to generic
	// params data type
	params_.fov = cv::Point2f(hFovRadians, hFovRadians * (float)height / (float)width);
	params_.fx = zedp.fx;
	params_.fy = zedp.fy;
	params_.cx = zedp.cx;
	params_.cy = zedp.cy;
}

#endif
