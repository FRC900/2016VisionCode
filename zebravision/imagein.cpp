#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "imagein.hpp"
#include "ZvSettings.hpp"

using namespace cv;

ImageIn::ImageIn(const char *inpath, ZvSettings *settings) :
  MediaIn(settings)
{
	imread(inpath).copyTo(_frame);
	if (_frame.empty())
		std::cerr << "Could not open image file " << inpath << std::endl;
	while (_frame.rows > 800)
		pyrDown(_frame, _frame);
}

bool ImageIn::isOpened(void) const
{
	return _frame.empty();
}

bool ImageIn::update(void)
{
	usleep(250000);
	return true;
}

bool ImageIn::getFrame(Mat &frame, Mat &depth, bool pause)
{
	(void)pause;
	if (_frame.empty())
		return false;
	frame = _frame.clone();
	depth = Mat();
	return true;
}

int ImageIn::width(void) const
{
	return _frame.cols;
}

int ImageIn::height(void) const
{
	return _frame.rows;
}

// Images have only 1 "frame"
int ImageIn::frameCount(void) const
{
	return 1;
}

int ImageIn::frameNumber(void) const
{
	return 1;
}
