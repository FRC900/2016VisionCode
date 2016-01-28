#include "zedin.hpp"

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
    		exit(-1);
	}
	_width = zed->getImageSize().width;
	_height = zed->getImageSize().height;
	stereoParams = *zed->getParameters();

	cv_frame.create(_height,_width,CV_8UC4);
	cv_normalDepth.create(_height,_width,CV_8UC4);
	cv_depth.create(_height,_width,CV_32F);
	cv_confidence.create(_height,_width,CV_8UC4);
}

ZedIn::ZedIn(char* svo_path)
{
	zed = new sl::zed::Camera(svo_path);
	// init computation mode of the zed
	sl::zed::ERRCODE err = zed->init(sl::zed::MODE::QUALITY, -1, true);
	cout << sl::zed::errcode2str(err) << endl;
	// Quit if an error occurred
	if (err != sl::zed::SUCCESS) {
    		delete zed;
    		exit(-1);
	}

	_width = zed->getImageSize().width;
	_height = zed->getImageSize().height;
	stereoParams = *zed->getParameters();
}

bool ZedIn::update()
{
	cout << 1 << endl;
	bool grabError = zed->grab(sl::zed::RAW);
	if (grabError) {
		cerr << "Error grabbing image from zed" << endl;
		return false;
	}

	if(_left)
		slMat2cvMat(zed->retrieveImage(sl::zed::SIDE::LEFT)).copyTo(cv_frame);
	else
		slMat2cvMat(zed->retrieveImage(sl::zed::SIDE::RIGHT)).copyTo(cv_frame);
	
	cout << 2 << endl;
	slMat2cvMat(zed->retrieveMeasure(sl::zed::MEASURE::DEPTH)).copyTo(cv_depth); //not normalized depth
	slMat2cvMat(zed->normalizeMeasure(sl::zed::MEASURE::DEPTH)).copyTo(cv_normalDepth); //normalized depth
	slMat2cvMat(zed->retrieveMeasure(sl::zed::MEASURE::CONFIDENCE)).copyTo(cv_confidence);

	cout << 3 << endl;
	cvtColor(cv_frame,cv_frame,CV_RGBA2RGB); //remove the alpha channel
	cout << 4 << endl;
   	return true;
}


double ZedIn::getDepthPoint(int x, int y) {
	float* data = (float*) cv_depth.data;
	float* ptr_image_num = (float*) ((int8_t*) data + y * cv_depth.step);
	float dist = ptr_image_num[x] / 1000.f;
	return dist;
}

