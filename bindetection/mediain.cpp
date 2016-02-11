#include "mediain.hpp"

MediaIn::MediaIn()
{
}

int MediaIn::frameCount(void) const
{
	return -1;
}

int MediaIn::frameCounter(void) const
{
	return -1;
}

void MediaIn::frameCounter(int frameCount)
{
	(void)frameCount;
}

bool MediaIn::getDepthMat(cv::Mat &depthMat)
{
	depthMat = Mat(); // return empty mat to indicate no depth info
	return false;     // in addition to returning false
}

sl::zed::CamParameters MediaIn::getCameraParams(bool left) {
	return sl::zed::CamParameters();
}

double MediaIn::getDepth(int x, int y)
{
	(void)x;
	(void)y;
	return -1000.;
}
