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

bool MediaIn::getDepthMat(Mat &depthMat) const
{
	depthMat = Mat(); // return empty mat to indicate no depth info
	return true;     // in addition to returning false
}

bool MediaIn::getNormDepthMat(Mat &normDepthMat) const
{
	normDepthMat = Mat(); // return empty mat to indicate no depth info
	return true;     // in addition to returning false
}

CameraParams MediaIn::getCameraParams(bool left) const
{
	(void)left;
	return CameraParams();
}

float MediaIn::getDepth(int x, int y)
{
	(void)x;
	(void)y;
	return -1000.;
}
