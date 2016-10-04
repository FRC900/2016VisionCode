#include <iostream>
#include "zedsvoin.hpp"
using namespace std;

#ifdef ZED_SUPPORT
using namespace cv;
using namespace sl::zed;

ZedSVOIn::ZedSVOIn(const char *inFileName, ZvSettings *settings) :
	ZedIn(settings)
{
	width_  = 0;
	height_ = 0;
	zed_ = new Camera(inFileName);

	if (zed_)
	{
		InitParams parameters;
		parameters.mode = PERFORMANCE;
		parameters.unit = MILLIMETER;
		parameters.verbose = 1;
		// init computation mode of the zed
		ERRCODE err = zed_->init(parameters);

		// Quit if an error occurred
		if (err != SUCCESS)
		{
			cout << errcode2str(err) << endl;
			delete zed_;
			zed_ = NULL;
		}
		else
		{
			//only for Jetson K1/X1 - see if it helps
			Camera::sticktoCPUCore(2);

			width_  = zed_->getImageSize().width;
			height_ = zed_->getImageSize().height;
		}
	}

	while (height_ > 700)
	{
		width_  /= 2;
		height_ /= 2;
	}
}

ZedSVOIn::~ZedSVOIn()
{
}

bool ZedSVOIn::isOpened(void) const
{
	return zed_ ? true : false;
}


bool ZedSVOIn::update(bool left)
{
	(void) left;
	// Do nothing for now - all the work is in getFrame
	if (zed_)
	{
		usleep (150000);
		return true;
	}
	return false;
}


bool ZedSVOIn::getFrame(cv::Mat &frame, cv::Mat &depth, bool pause)
{
	if (!zed_)
		return false;

	if (!pause)
	{
		if (zed_->grab())
			return false;

		slFrame_ = zed_->retrieveImage(left ? SIDE::LEFT : SIDE::RIGHT);
		slDepth_ = zed_->retrieveMeasure(MEASURE::DEPTH);
		setTimeStamp();
		incFrameNumber();
		cvtColor(slMat2cvMat(slFrame_), frame_, CV_RGBA2RGB);
		slMat2cvMat(slDepth_).copyTo(depthMat_);

		while (frame_.rows > 700)
		{
			pyrDown(frame_, frame_);
			pyrDown(depthMat_, depthMat_);
		}
	}
	frame_.copyTo(frame);
	depthMat_.copyTo(depth);

	return true;
}


bool ZedSVOIn::update(void)
{
	return update(true);
}


int ZedSVOIn::frameCount(void) const
{
	// Luckily getSVONumberOfFrames() returns -1 if we're
	// capturing from a camera, which is also what the rest
	// of our code expects in that case
	if (zed_)
		return zed_->getSVONumberOfFrames();

	// If using zms or a live camera, there's no way to tell
	return -1;
}


// Seek to a given frame number. This is possible if the
// input is a video. If reading live camera data it will
// fail, but nothing we can do about that so fail silently
void ZedSVOIn::frameNumber(int frameNumber)
{
	if (zed_ && zed_->setSVOPosition(frameNumber))
		setFrameNumber(frameNumber);
}


#else
ZedSVOIn::ZedSVOIn(const char *inFileName, ZvSettings *settings) :
	MediaIn(settings)
{
	(void)inFileName;
	cerr << "Zed support not compiled in" << endl;
}


ZedSVOIn::~ZedSVOIn()
{
}
#endif
