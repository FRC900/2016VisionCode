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
	(void)depthMat;
	return false;
}

double MediaIn::getDepth(int x, int y)
{
	(void)x;
	(void)y;
   return -1000.;
}