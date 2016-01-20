#include "zedin.hpp"

#ifdef ZED_SUPPORT
//cuda include
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "npp.h"
#include "device_functions.h"

using namespace cv;
using namespace std;


ZedIn::ZedIn()
{
	zed = new sl::zed::Camera(sl::zed::VGA);
	// init computation mode of the zed
	sl::zed::ERRCODE err = zed->init(sl::zed::MODE::QUALITY, -1, true);
	cout << sl::zed::errcode2str(err) << endl;
	// Quit if an error occurred
	if (err != sl::zed::SUCCESS) {
		delete zed;
		zed = NULL;
	}else{
		width_ = zed->getImageSize().width;
		height_ = zed->getImageSize().height;
	}
}

bool ZedIn::getNextFrame(cv::Mat &frame,bool left,bool pause)
{
	cv::Mat imageCPU(height_, width_, CV_8UC4);
	if(zed == NULL)
	{
		return false;
	}
	if(pause == false)
	{
		zed->grab(sl::zed::RAW);
		if(left) 
		{
			imageGPU = zed->getView_gpu(sl::zed::STEREO_LEFT);
		}else{
			imageGPU = zed->getView_gpu(sl::zed::STEREO_RIGHT);
		}
		depthMat = zed->retrieveMeasure(sl::zed::MEASURE::DEPTH);
		cudaMemcpy2D((uchar*) imageCPU.data, imageCPU.step, (Npp8u*) imageGPU.data, imageGPU.step, imageGPU.getWidthByte(), imageGPU.height, cudaMemcpyDeviceToHost);
		cvtColor(imageCPU,imageCPU,CV_RGBA2RGB);
	}
	frame = imageCPU.clone();
	return true;
}

bool ZedIn::getNextFrame(cv::Mat &frame,bool pause) {
	return getNextFrame(frame,true, pause);
}



double ZedIn::getDepth(int x, int y) {
	//zed->grab(sl::zed::FULL);
	float* data = (float*) depthMat.data;
	float* ptr_image_num = (float*) ((int8_t*) data + y * depthMat.step);
	float dist = ptr_image_num[x] / 1000.f;
	return dist;
}


int ZedIn::height(void)
{
	return height_;
}

int ZedIn::width(void)
{
	return width_;
}
#else
ZedIn::ZedIn()
{
}

bool ZedIn::getNextFrame(cv::Mat &frame,bool pause) 
{
	return false;
}

int ZedIn::height(void)
{
	return 0;
}

int ZedIn::width(void)
{
	return 0;
}
#endif
