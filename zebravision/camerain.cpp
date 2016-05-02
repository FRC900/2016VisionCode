#include <iostream>
#include <opencv2/opencv.hpp>

#include "camerain.hpp"

using namespace cv;
using namespace std;

CameraIn::CameraIn(int stream, ZvSettings *settings) :
  MediaIn(settings),
	frameNumber_(0),
	width_(1280),
  height_(720),
	fps_(30.),
	cap_(stream)
{
	if (cap_.isOpened())
	{
    if (!loadSettings()) {
      cerr << "Failed to load CameraIn settings" << endl;
    }

		cap_.set(CV_CAP_PROP_FPS, fps_);
		cap_.set(CV_CAP_PROP_FRAME_WIDTH, width_);
		cap_.set(CV_CAP_PROP_FRAME_HEIGHT, height_);
		// getNextFrame resizes large inputs,
		// make sure width and height match
		while (height_ > 700)
		{
			width_ /= 2;
			height_ /= 2;
		}
	}
	else
		std::cerr << "Could not open camera" << std::endl;
}

CameraIn::~CameraIn()
{
	//FileStorage fs("camerain.xml", FileStorage::WRITE);
	// Can't get FPS from a camera even though it
	// can be set...
	//fs << "camerain" << "{" ;
	//fs << "fps" << fps_;
	//fs << "width" << cap_.get(CV_CAP_PROP_FRAME_WIDTH);
	//fs << "height" << cap_.get(CV_CAP_PROP_FRAME_HEIGHT);
	//fs << "}";
}

bool CameraIn::loadSettings()
{
  if (_settings) {
    _settings->getDouble(getClassName(), "fps", fps_);
    _settings->getInt(getClassName(), "width", width_);
    _settings->getInt(getClassName(), "height", height_);
cerr << "ESB: fps=" << fps_ << " width=" << width_ << " height=" << height_ << endl;
    return true;
  }
	return false;
}

bool CameraIn::saveSettings()
{
	//XMLElement* root = doc.FirstChildElement();
	//XMLElement* newElement = doc.NewElement( "Subelement" );
  //root->InsertEndChild( newElement );

	return true;
}

bool CameraIn::isOpened() const
{
	return cap_.isOpened();
}

bool CameraIn::update(void)
{
	if (!cap_.isOpened()  ||
	    !cap_.grab() ||
	    !cap_.retrieve(localFrame_))
		return false;
	boost::lock_guard<boost::mutex> guard(_mtx);
	setTimeStamp();
	localFrame_.copyTo(_frame);
	while (_frame.rows > 700)
		pyrDown(_frame, _frame);
	frameNumber_ += 1;
	return true;
}

bool CameraIn::getFrame(Mat &frame, Mat &depth, bool pause)
{
	(void)pause;
	if (!cap_.isOpened())
		return false;
	depth = Mat();
	boost::lock_guard<boost::mutex> guard(_mtx);
	if (_frame.empty())
		return false;
	_frame.copyTo(frame);
	lockedFrameNumber_ = frameNumber_;
	lockedTimeStamp_ = timeStamp_;
	return true;
}

int CameraIn::width(void) const
{
   return width_;
}

int CameraIn::height(void) const
{
   return height_;
}

int CameraIn::frameNumber(void) const
{
   return lockedFrameNumber_;
}
