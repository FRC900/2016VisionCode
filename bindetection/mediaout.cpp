#include "mediaout.hpp"

using namespace cv;

MediaOut::MediaOut(void)
{
}

MediaOut::~MediaOut(void)
{
}

bool MediaOut::saveFrame(const Mat &frame, const Mat &depth)
{
	(void)frame;
	(void)depth;
	return false;
}
