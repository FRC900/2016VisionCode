#include <iostream>
#include "zedin.hpp"
using namespace std;

#ifdef ZED_SUPPORT
#include <boost/filesystem.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cvMatSerialize.hpp"

using namespace cv;
using namespace boost::filesystem;

void brightnessCallback(int value, void *data);
void contrastCallback(int value, void *data);
void hueCallback(int value, void *data);
void saturationCallback(int value, void *data);
void gainCallback(int value, void *data);
void whiteBalanceTemperatureCallback(int value, void *data);

ZedIn::ZedIn(const char *inFileName, const char *outFileName, bool gui) :
		zed_(NULL),
		width_(0),
		height_(0),
	frameNumber_(0),
	serializeIn_(NULL), 
	archiveIn_(NULL),
	serializeOut_(NULL),
	archiveOut_(NULL)
#if 0
	,
	serializeFrameSize_(0)
#endif
{
	if (inFileName)
	{
		// Might be svo, might be zms
		string fnExt = path(inFileName).extension().string();
		if ((fnExt == ".svo") || (fnExt == ".SVO"))
			zed_ = new sl::zed::Camera(inFileName);
		else if ((fnExt == ".zms") || (fnExt == ".ZMS"))
		{
			// ZMS file is home-brewed serialization format
			// which just dumps raw a image and depth Mat data to a file.  
			// Apply a light bit of compression because
			// the files will get out of hand quickly otherwise
			serializeIn_ = new ifstream(inFileName, ios::in | ios::binary);
			if (serializeIn_ && serializeIn_->is_open())
			{
				cerr << "Loading " << inFileName << " for reading" << endl;

				filtSBIn_.push(boost::iostreams::zlib_decompressor());
				filtSBIn_.push(*serializeIn_);
				archiveIn_ = new boost::archive::binary_iarchive(filtSBIn_);
			}
			else
			{
				cerr << "Zed init : Could not open " << inFileName << " for reading" << endl;
				deletePointers();
			}
		}
		else
			cerr << "Zed failed to start : unknown file extension " << fnExt << endl;
	}
	else // Open an actual camera for input
		zed_ = new sl::zed::Camera(sl::zed::VGA, 30);

	// Save the raw camera stream to disk.  This uses a home-brew
	// method to serialize image and depth data to disk rather than
	// relying on Stereolab's SVO format.
	if (outFileName)
	{
		serializeOut_ = new ofstream(outFileName, ios::out | ios::binary);
		if (serializeOut_ && serializeOut_->is_open())
		{
			filtSBOut_.push(boost::iostreams::zlib_compressor(boost::iostreams::zlib::best_speed));
			filtSBOut_.push(*serializeOut_);
			archiveOut_ = new boost::archive::binary_oarchive(filtSBOut_);
		}
		else
		{
			cerr << "Zed init : could not open output file " << outFileName << endl;
			// Don't want to close the input stream since we
			// still want to run even if we can't capture for
			// whatever reason (disk full, etc)
			if (serializeOut_)
				delete serializeOut_;
			serializeOut_ = NULL;
		}
	}

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
			width_  = zed_->getImageSize().width;
			height_ = zed_->getImageSize().height;

#if 0
			brightness_ = zed_->getCameraSettingsValue(sl::zed::ZED_BRIGHTNESS);
			contrast_ = zed_->getCameraSettingsValue(sl::zed::ZED_CONTRAST);
			hue_ = zed_->getCameraSettingsValue(sl::zed::ZED_HUE);
			saturation_ = zed_->getCameraSettingsValue(sl::zed::ZED_SATURATION);
			gain_ = zed_->getCameraSettingsValue(sl::zed::ZED_GAIN);
			whiteBalance_ = zed_->getCameraSettingsValue(sl::zed::ZED_WHITEBALANCE);
#endif
			brightness_ = 5;
			contrast_ = 5;
			hue_ = 6;
			saturation_ = 3;
			gain_ = 2;
			whiteBalance_ = 3100;

			cout << "brightness_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_BRIGHTNESS) << endl;
			cout << "contrast_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_CONTRAST) << endl;
			cout << "hue_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_HUE) << endl;
			cout << "saturation_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_SATURATION) << endl;
			cout << "gain_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_GAIN) << endl;
			cout << "whiteBalance_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_WHITEBALANCE) << endl;
if (gui)
{
		cv::namedWindow("Adjustments", CV_WINDOW_NORMAL);
		cv::createTrackbar("Brightness", "Adjustments", &brightness_, 9, brightnessCallback, this);
		cv::createTrackbar("Contrast", "Adjustments", &contrast_, 9, contrastCallback, this);
		cv::createTrackbar("Hue", "Adjustments", &hue_, 12, hueCallback, this);
		cv::createTrackbar("Saturation", "Adjustments", &saturation_, 9, saturationCallback, this);
		cv::createTrackbar("Gain", "Adjustments", &gain_, 9, gainCallback, this);
		cv::createTrackbar("White Balance Temperature", "Adjustments", &whiteBalance_, 6501, whiteBalanceTemperatureCallback, this);
}
		}
	}
	else if (serializeIn_ && serializeIn_->is_open())
	{
		// Zed == NULL and serializeStream_ means reading from 
		// a serialized file. Grab height_ and width_
		// Also figure out how big a frame is so we can
		// use random access to get at any frame
		*archiveIn_ >> frame_ >> depthMat_;
		frameNumber_ += 1;
#if 0
		serializeFrameSize_ = serializeIn_->tellg();
#endif
		width_  = frame_.cols;
		height_ = frame_.rows;
		
#if 0
		// Jump back to start of file
		serializeIn_->clear();
		serializeIn_->seekg(0);
#endif
	}
	while (height_ > 800)
	{
		width_  /= 2;
		height_ /= 2;
	}
}


void ZedIn::deletePointers(void)
{
	if (archiveIn_)
	{
		delete archiveIn_;
		archiveIn_ = NULL;
	}
	if (archiveOut_)
	{
		delete archiveOut_;
		archiveOut_ = NULL;
	}
	if (serializeIn_)
	{
		delete serializeIn_;
		serializeIn_ = NULL;
	}
	if (serializeOut_)
	{
		delete serializeOut_;
		serializeOut_ = NULL;
	}
}


ZedIn::~ZedIn()
{
	if (zed_)
		delete zed_;
}


bool ZedIn::getNextFrame(Mat &frame, bool left, bool pause)
{
	if ((zed_ == NULL) && 
		!(serializeIn_ && serializeIn_->is_open() && archiveIn_))
		return false;

	if (pause == false)
	{
		// Read from either the zed camera or from 
		// a previously-serialized ZMS file
		if (zed_)
		{
			cout << "brightness_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_BRIGHTNESS) << endl;
			cout << "contrast_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_CONTRAST) << endl;
			cout << "hue_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_HUE) << endl;
			cout << "saturation_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_SATURATION) << endl;
			cout << "gain_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_GAIN) << endl;
			cout << "whiteBalance_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_WHITEBALANCE) << endl;
			zed_->grab(sl::zed::RAW);

			slMat2cvMat(zed_->retrieveImage(left ? sl::zed::SIDE::LEFT : sl::zed::SIDE::RIGHT)).copyTo(frameRGBA_);
			cvtColor(frameRGBA_, frame_, CV_RGBA2RGB);

			slMat2cvMat(zed_->retrieveMeasure(sl::zed::MEASURE::DEPTH)).copyTo(depthMat_); //not normalized depth
		}
		else if (serializeIn_ && serializeIn_->is_open())
		{
			// Ugly try-catch to detect EOF
			try
			{
				*archiveIn_ >> frame_ >> depthMat_;
			}
			catch (const boost::archive::archive_exception &e) 
			{
				return false;
			}
		}

		// Write output to serialized file if it is open
		if (serializeOut_ && serializeOut_->is_open())
			*archiveOut_ << frame_ << depthMat_;

		while (frame_.rows > 800)
		{
			pyrDown(frame_, frame_);
			pyrDown(depthMat_, depthMat_);
		}
		frameNumber_ += 1;
	}
	frame = frame_.clone();
	return true;
}


bool ZedIn::getNextFrame(Mat &frame, bool pause) 
{
	return getNextFrame(frame, true, pause);
}


int ZedIn::frameCount(void) const
{
#if 0
	// If we're using an input file we can calculate this.
	// If using a video, there's no way to tell
	if (serializeIn_ && serializeIn_->is_open() && serializeFrameSize_)
		return (serializeIn_->tellg() / serializeFrameSize_);
#endif

	// Luckily getSVONumberOfFrames() returns -1 if we're
	// capturing from a camera, which is also what the rest
	// of our code expects in that case
	if (zed_)
		return zed_->getSVONumberOfFrames();
		
	return -1;
}


int ZedIn::frameNumber(void) const
{
	return frameNumber_;
}


// Seek to a given frame number. This is possible if the
// input is a video. If reading live camera data it will
// fail, but nothing we can do about that so fail silently
void ZedIn::frameNumber(int frameNumber)
{
#if 0
	if (serializeIn_ && serializeIn_->is_open() && serializeFrameSize_)
	{
		serializeIn_->seekg(frameCount * serializeFrameSize_);
		frameNumber_ = frameNumber;
	}
	else 
#endif
	if (zed_)  

	{
		if (zed_->setSVOPosition(frameNumber))
			frameNumber_ = frameNumber;
	}
}


double ZedIn::getDepth(int x, int y) 
{
	float* ptr_image_num = (float*) ((int8_t*)depthMat_.data + y * depthMat_.step);
	return ptr_image_num[x];
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

sl::zed::CamParameters ZedIn::getCameraParams(bool left) const
{
	if (zed_)
	{
		if(left)
			return (zed_->getParameters())->LeftCam;
		return (zed_->getParameters())->RightCam;
	}
	// Take a guess based on one of our cameras
	sl::zed::CamParameters params;
	params.fx = 720;
	params.fy = 720;
	params.cx = height_/2;
	params.cy = width_/2;
	return params;
}

void brightnessCallback(int value, void *data)
{
    ZedIn *zedPtr = (ZedIn *)data;
	zedPtr->brightness_ = value;
	if (zedPtr->zed_)
	{
		zedPtr->zed_->setCameraSettingsValue(sl::zed::ZED_BRIGHTNESS, value);
	}
}
void contrastCallback(int value, void *data)
{
    ZedIn *zedPtr = (ZedIn *)data;
	zedPtr->contrast_ = value;
	if (zedPtr->zed_)
	{
		zedPtr->zed_->setCameraSettingsValue(sl::zed::ZED_CONTRAST, value);
	}
}
void hueCallback(int value, void *data)
{
    ZedIn *zedPtr = (ZedIn *)data;
	zedPtr->hue_ = value;
	if (zedPtr->zed_)
	{
		zedPtr->zed_->setCameraSettingsValue(sl::zed::ZED_HUE, value);
	}
}
void saturationCallback(int value, void *data)
{
    ZedIn *zedPtr = (ZedIn *)data;
	zedPtr->saturation_ = value;
	if (zedPtr->zed_)
	{
		zedPtr->zed_->setCameraSettingsValue(sl::zed::ZED_SATURATION, value);
	}
}
void gainCallback(int value, void *data)
{
    ZedIn *zedPtr = (ZedIn *)data;
	zedPtr->gain_ = value;
	if (zedPtr->zed_)
	{
		zedPtr->zed_->setCameraSettingsValue(sl::zed::ZED_GAIN, value);
	}
}
void whiteBalanceTemperatureCallback(int value, void *data)
{
    ZedIn *zedPtr = (ZedIn *)data;
	zedPtr->whiteBalance_ = value;
	if (zedPtr->zed_)
	{
		zedPtr->zed_->setCameraSettingsValue(sl::zed::ZED_WHITEBALANCE, value);
	}
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


ZedIn::ZedIn(const char *filename, const char *outputName)
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
