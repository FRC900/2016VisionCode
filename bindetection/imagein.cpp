#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "imagein.hpp"

using namespace cv;

ImageIn::ImageIn(const char *inpath, const char *outpath)
{
	imread(inpath).copyTo(_frame);
	if (_frame.empty())
		std::cerr << "Could not open image file " << inpath << std::endl;
	while (_frame.rows > 800)
		pyrDown(_frame, _frame);
	outpath_ = outpath;
}

bool ImageIn::update() {
	usleep(500000);
	return true;
}

bool ImageIn::getFrame(Mat &frame, Mat &depth)
{
	if (_frame.empty())
		return false;
	frame = _frame.clone();
	depth = Mat();
	return true;
}

bool ImageIn::saveFrame(Mat &frame, Mat &depth) {
	//strip the file extension and replace it with png because we're saving an image
	std::stringstream ss;
	size_t lastindex = outpath_.find_last_of(".");
	ss << outpath_.substr(0,lastindex);
	ss << ".png";
	imwrite(ss.str(), frame);
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
