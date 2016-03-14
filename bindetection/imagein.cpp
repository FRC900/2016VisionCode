#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "imagein.hpp"

using namespace cv;

ImageIn::ImageIn(char *inpath, char *outpath)
{
	//set isvideo to true to make sure that the grab loop doesn't run obscenely fast
	isVideo = true;
	imread(inpath).copyTo(_frame);
	if (_frame.empty())
		std::cerr << "Could not open image file " << inpath << std::endl;
	while (_frame.rows > 800)
		pyrDown(_frame, _frame);
	outpath_ = outpath;
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

bool ImageIn::saveFrame(Mat &frame) {
	imwrite(outpath_, frame);
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
