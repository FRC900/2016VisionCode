#include <iostream>
#include "zedin.hpp"
using namespace std;

#ifdef ZED_SUPPORT

using namespace cv;

ZedIn::ZedIn(const char *filename)
{
	if (filename)
		zed_ = new sl::zed::Camera(filename);
	else
		zed_ = new sl::zed::Camera(sl::zed::VGA, 30);

	if (zed_)
	{
		// init computation mode of the zed
		sl::zed::ERRCODE err = zed_->init(sl::zed::MODE::QUALITY, -1, true);
		cout << sl::zed::errcode2str(err) << endl;
		// Quit if an error occurred
		if (err != sl::zed::SUCCESS) 
		{
			delete zed_;
			zed_ = NULL;
		}
		else
		{
			width_     = zed_->getImageSize().width;
			height_    = zed_->getImageSize().height;
			frameRGBA_ = Mat(height_, width_, CV_8UC4);
			frameCounter_ = 0;
		}
	}
}

ZedIn::~ZedIn()
{
	if (zed_)
		delete zed_;
}


bool ZedIn::getNextFrame(Mat &frame, bool left, bool pause)
{
	if (zed_ == NULL)
		return false;
	if (pause == false)
	{
		zed_->grab(sl::zed::RAW);

		slMat2cvMat(zed_->retrieveImage(left ? sl::zed::SIDE::LEFT : sl::zed::SIDE::RIGHT)).copyTo(frameRGBA_);
		cvtColor(frameRGBA_, frame_, CV_RGBA2RGB);

		slMat2cvMat(zed_->retrieveMeasure(sl::zed::MEASURE::DEPTH)).copyTo(depthMat_); //not normalized depth

		frameCounter_ += 1;
	}
	frame = frame_.clone();
	return true;
}

bool ZedIn::getNextFrame(Mat &frame, bool pause) 
{
	return getNextFrame(frame, true, pause);
}

double ZedIn::getDepth(int x, int y) 
{
	float* ptr_image_num = (float*) ((int8_t*)depthMat_.data + y * depthMat_.step);
	return ptr_image_num[x] / 1000.f;
}

bool ZedIn::getDepthMat(Mat &depthMat)
{
	depthMat_.copyTo(depthMat); //not normalized depth
	return true;
}

int ZedIn::width(void) const
{
	return width_;
}

int ZedIn::height(void) const
{
	return height_;
}

#else

ZedIn::~ZedIn()
{
}

int ZedIn::width(void) const
{
	return 0;
}

int ZedIn::height(void) const
{
	return 0;
}

ZedIn::ZedIn(const char *filename)
{
	(void)filename;
	cerr << "Zed support not compiled in" << endl;
}

bool ZedIn::getNextFrame(Mat &frame, bool pause) 
{
	(void)frame;
	(void)pause;
	return false;
}

#endif
