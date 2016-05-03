#include <sys/time.h>

#include "mediain.hpp"
#include "ZvSettings.hpp"

using namespace cv;

MediaIn::MediaIn(ZvSettings *settings) :
	settings_(settings),
	lockedTimeStamp_(0)
{
}

bool MediaIn::loadSettings()
{
	// MediaIn has no settings to load currently
	return true;
}

bool MediaIn::saveSettings() const
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

long long MediaIn::timeStamp(void) const
{
	return lockedTimeStamp_;
}

long long MediaIn::setTimeStamp(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);

	timeStamp_ = (long long)tv.tv_sec * 1000000000ULL +
		         (long long)tv.tv_usec * 1000ULL;

	return timeStamp_;
}

CameraParams MediaIn::getCameraParams(bool left) const
{
	(void)left;
	return CameraParams();
}
