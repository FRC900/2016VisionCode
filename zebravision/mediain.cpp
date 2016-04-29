#include "mediain.hpp"
#include "ZvSettings.hpp"

using namespace cv;

MediaIn::MediaIn(ZvSettings *settings) : _settings(settings)
{
}

bool MediaIn::loadSettings()
{
	// MediaIn has no settings to load currently
	return true;
}

bool MediaIn::saveSettings()
{
	// MediaIn has no settings to save currently
	return true;
}

bool MediaIn::isOpened(void) const
{
	return false;
}

int MediaIn::frameCount(void) const
{
	return -1;
}

int MediaIn::frameNumber(void) const
{
	return -1;
}

void MediaIn::frameNumber(int frameNumber)
{
	(void)frameNumber;
}

CameraParams MediaIn::getCameraParams(bool left) const
{
	(void)left;
	return CameraParams();
}
