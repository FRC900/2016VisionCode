#include "mediain.hpp"

using namespace cv;

MediaIn::MediaIn()
{
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
