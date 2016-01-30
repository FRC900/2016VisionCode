#include <iostream>
#include "zedin.hpp"
using namespace std;

#ifdef ZED_SUPPORT
//cuda include
#include "npp.h"

using namespace cv;

ZedIn::ZedIn(const char *filename)
{
	if (filename)
		zed = new sl::zed::Camera(filename);
	else
		zed = new sl::zed::Camera(sl::zed::VGA);

	if (zed)
	{
		// init computation mode of the zed
		sl::zed::ERRCODE err = zed->init(sl::zed::MODE::QUALITY, -1, true);
		cout << sl::zed::errcode2str(err) << endl;
		// Quit if an error occurred
		if (err != sl::zed::SUCCESS) 
		{
			delete zed;
			zed = NULL;
		}
		else
		{
			width_     = zed->getImageSize().width;
			height_    = zed->getImageSize().height;
			frameRGBA_ = cv::Mat(height_, width_, CV_8UC4);
			frameCounter_ = 0;
		}
	}
}

bool ZedIn::getNextFrame(cv::Mat &frame, bool left, bool pause)
{
	if(zed == NULL)
		return false;
	if (pause == false)
	{
		zed->grab(sl::zed::RAW);
		imageGPU = zed->getView_gpu(left ? sl::zed::STEREO_LEFT : sl::zed::STEREO_RIGHT);
		depthMat = zed->retrieveMeasure(sl::zed::MEASURE::DEPTH);
		cudaMemcpy2D((uchar*) frameRGBA_.data, frameRGBA_.step, (Npp8u*) imageGPU.data, imageGPU.step, imageGPU.getWidthByte(), imageGPU.height, cudaMemcpyDeviceToHost);
		cvtColor(frameRGBA_, frame_, CV_RGBA2RGB);
		frameCounter_ += 1;
	}
	frame = frame_.clone();
	return true;
}

bool ZedIn::getNextFrame(cv::Mat &frame, bool pause) 
{
	return getNextFrame(frame, true, pause);
}

double ZedIn::getDepth(int x, int y) 
{
	//zed->grab(sl::zed::FULL);
	float* data = (float*) depthMat.data;
	float* ptr_image_num = (float*) ((int8_t*) data + y * depthMat.step);
	float dist = ptr_image_num[x] / 1000.f;
	return dist;
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

ZedIn::ZedIn(const char *filename)
{
	cerr << "Zed support not compiled in" << endl;
}

bool ZedIn::getNextFrame(cv::Mat &frame, bool pause) 
{
	return false;
}

int ZedIn::width(void) const
{
		return 0;
}

int ZedIn::height(void) const
{
		return 0;
}

#endif
