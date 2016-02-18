#include "mediain.hpp"

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

bool MediaIn::getDepthMat(cv::Mat &depthMat)
{
	depthMat = Mat(); // return empty mat to indicate no depth info
	return false;     // in addition to returning false
}

double MediaIn::getDepth(int x, int y)
{
	(void)x;
	(void)y;
	return -1000.;
}
