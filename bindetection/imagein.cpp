#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "imagein.hpp"

using namespace cv;

ImageIn::ImageIn(const char *path)
{
	//set isvideo to true to make sure that the grab loop doesn't run obscenely fast
	isVideo = true;
	imread(path).copyTo(_frame);
	if (_frame.empty())
		std::cerr << "Could not open image file " << path << std::endl;
	while (_frame.rows > 800)
		pyrDown(_frame, _frame);
}

bool ImageIn::update() {
	boost::lock_guard<boost::mutex> guard(_mtx);
	if (_frame.empty())
		return false;
	return true;
}

bool ImageIn::getFrame(Mat &frame)
{
	boost::lock_guard<boost::mutex> guard(_mtx);
	if (_frame.empty())
		return false;
	frame = _frame.clone();
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
