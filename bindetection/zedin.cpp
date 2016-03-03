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

void zedBrightnessCallback(int value, void *data);
void zedContrastCallback(int value, void *data);
void zedHueCallback(int value, void *data);
void zedSaturationCallback(int value, void *data);
void zedGainCallback(int value, void *data);
void zedWhiteBalanceCallback(int value, void *data);

ZedIn::ZedIn(const char *inFileName, const char *outFileName, bool gui) :
	zed_(NULL),
	width_(0),
	height_(0),
	frameNumber_(0),
	serializeIn_(NULL), 
	filtSBIn_(NULL),
	archiveIn_(NULL),
	serializeOut_(NULL),
	filtSBOut_(NULL),
	archiveOut_(NULL) ,
	serializeFrameStart_(0),
	serializeFrameSize_(0)
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
			cerr << "Loading " << inFileName << " for reading" << endl;
			if (!openSerializeInput(inFileName))
				cerr << "Zed init : Could not open " << inFileName << " for reading" << endl;
		}
		else
			cerr << "Zed failed to start : unknown file extension " << fnExt << endl;
	}
	else // Open an actual camera for input
		zed_ = new sl::zed::Camera(sl::zed::HD720);

	// Save the raw camera stream to disk.  This uses a home-brew
	// method to serialize image and depth data to disk rather than
	// relying on Stereolab's SVO format.
	if (outFileName)
	{
		outFileName_ = outFileName;
		if (!openSerializeOutput(outFileName_.c_str()))
			cerr << "Zed init : could not open output file " << outFileName << endl;
	}

	if (zed_)
	{
		// init computation mode of the zed
		sl::zed::ERRCODE err = zed_->init(sl::zed::MODE::PERFORMANCE, -1, true);
		// Quit if an error occurred
		if (err != sl::zed::SUCCESS) 
		{
			cout << sl::zed::errcode2str(err) << endl;
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
			zedBrightnessCallback(3, this);
			zedContrastCallback(5, this);
			zedHueCallback(6, this);
			zedSaturationCallback(3, this);
			zedGainCallback(1, this);
			zedWhiteBalanceCallback(3100, this);

			cout << "brightness_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_BRIGHTNESS) << endl;
			cout << "contrast_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_CONTRAST) << endl;
			cout << "hue_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_HUE) << endl;
			cout << "saturation_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_SATURATION) << endl;
			cout << "gain_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_GAIN) << endl;
			cout << "whiteBalance_ = " << zed_->getCameraSettingsValue(sl::zed::ZED_WHITEBALANCE) << endl;
			if (gui)
			{
				cv::namedWindow("Adjustments", CV_WINDOW_NORMAL);
				cv::createTrackbar("Brightness", "Adjustments", &brightness_, 9, zedBrightnessCallback, this);
				cv::createTrackbar("Contrast", "Adjustments", &contrast_, 9, zedContrastCallback, this);
				cv::createTrackbar("Hue", "Adjustments", &hue_, 12, zedHueCallback, this);
				cv::createTrackbar("Saturation", "Adjustments", &saturation_, 9, zedSaturationCallback, this);
				cv::createTrackbar("Gain", "Adjustments", &gain_, 9, zedGainCallback, this);
				cv::createTrackbar("White Balance", "Adjustments", &whiteBalance_, 6501, zedWhiteBalanceCallback, this);
			}
		}
	}
	else if (archiveIn_)
	{
		// Zed == NULL and serializeStream_ means reading from 
		// a serialized file. Grab height_ and width_
#if 0
		// Also figure out how big a frame is so we can
		// use random access to get at any frame
		serializeFrameStart_ = serializeIn_->tellg();
#endif
		*archiveIn_ >> frame_ >> depthMat_;
		frameNumber_ += 1;
#if 0
		serializeFrameSize_ = serializeIn_->tellg() - serializeFrameStart_;
		if (!openSerializeInput(inFileName))
			cerr << "Zed init : Could not reopen " << inFileName << " for reading" << endl;
#endif
		width_  = frame_.cols;
		height_ = frame_.rows;
	}
	while (height_ > 700)
	{
		width_  /= 2;
		height_ /= 2;
	}
}

// Input needs 3 things. First is a standard ifstream to read from
// Next is an (optional) filtered stream buffer. This is used to
// uncompress on the fly - uncompressed files take up way too
// much space. Last item is the actual boost binary archive template
// If all three are opened, return true. If not, delete and set to
// NULL all pointers related to serialized Input
bool ZedIn::openSerializeInput(const char *inFileName)
{
	deleteInputPointers();
	serializeIn_ = new ifstream(inFileName, ios::in | ios::binary);
	if (!serializeIn_ || !serializeIn_->is_open())
	{
		cerr << "Could not open ifstream(" << inFileName << ")" << endl;
		deleteInputPointers();
		return false;
	}

	filtSBIn_= new boost::iostreams::filtering_streambuf<boost::iostreams::input>;
	if (!filtSBIn_)
	{
		cerr << "Could not create filtering_streambuf<input>" << endl;
		deleteInputPointers();
		return false;
	}
	filtSBIn_->push(boost::iostreams::zlib_decompressor());
	filtSBIn_->push(*serializeIn_);
	archiveIn_ = new boost::archive::binary_iarchive(*filtSBIn_);
	if (!archiveIn_)
	{
		cerr << "Could not create new binary_iarchive" << endl;
		deleteInputPointers();
		return false;
	}
	return true;
}

// Output needs 3 things. First is a standard ofstream to write to
// Next is an (optional) filtered stream buffer. This is used to
// compress on the fly - uncompressed files take up way too
// much space. Last item is the actual boost binary archive template
// If all three are opened, return true. If not, delete and set to
// NULL all pointers related to serialized Output
bool ZedIn::openSerializeOutput(const char *outFileName)
{
	deleteOutputPointers();
	serializeOut_ = new ofstream(outFileName, ios::out | ios::binary);
	if (!serializeOut_ || !serializeOut_->is_open())
	{
		cerr << "Could not open ofstream(" << outFileName << ")" << endl;
		deleteOutputPointers();
		return false;
	}
	filtSBOut_= new boost::iostreams::filtering_streambuf<boost::iostreams::output>;
	if (!filtSBOut_)
	{
		cerr << "Could not create filtering_streambuf<output> in constructor" <<endl;
		deleteOutputPointers();
		return false;
	}
	filtSBOut_->push(boost::iostreams::zlib_compressor(boost::iostreams::zlib::best_speed));
	filtSBOut_->push(*serializeOut_);
	archiveOut_ = new boost::archive::binary_oarchive(*filtSBOut_);
	if (!archiveOut_)
	{
		cerr << "Could not create binary_oarchive in constructor" <<endl;
		deleteOutputPointers();
		return false;
	}
	return true;
}

// Helper to easily delete and NULL out input file pointers
void ZedIn::deleteInputPointers(void)
{
	if (archiveIn_)
	{
		delete archiveIn_;
		archiveIn_ = NULL;
	}
	if (filtSBIn_)
	{
		delete filtSBIn_;
		filtSBIn_ = NULL;
	}
	if (serializeIn_)
	{
		delete serializeIn_;
		serializeIn_ = NULL;
	}
}

// Helper to easily delete and NULL out output file pointers
void ZedIn::deleteOutputPointers(void)
{
	if (archiveOut_)
	{
		delete archiveOut_;
		archiveOut_ = NULL;
	}
	if (filtSBOut_)
	{
		delete filtSBOut_;
		filtSBOut_ = NULL;
	}
	if (serializeOut_)
	{
		delete serializeOut_;
		serializeOut_ = NULL;
	}
}


void ZedIn::deletePointers(void)
{
	deleteInputPointers();
	deleteOutputPointers();
}


ZedIn::~ZedIn()
{
	deletePointers();
	if (zed_)
		delete zed_;
}


bool ZedIn::getNextFrame(Mat &frame, bool left, bool pause)
{
	if ((zed_ == NULL) && (archiveIn_ == NULL))
		return false;

	if (pause == false)
	{
		// Read from either the zed camera or from 
		// a previously-serialized ZMS file
		if (zed_)
		{
			zed_->grab(sl::zed::RAW);

			slMat2cvMat(zed_->retrieveImage(left ? sl::zed::SIDE::LEFT : sl::zed::SIDE::RIGHT)).copyTo(frameRGBA_);
			slMat2cvMat(zed_->retrieveMeasure(sl::zed::MEASURE::DEPTH)).copyTo(depthMat_); //not normalized depth
			slMat2cvMat(zed_->normalizeMeasure(sl::zed::MEASURE::DEPTH)).copyTo(normDepthMat_);

			cvtColor(frameRGBA_, frame_, CV_RGBA2RGB);
		}
		else if (archiveIn_)
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
			normalize(depthMat_, normDepthMat_, 0, 255, NORM_MINMAX, CV_8UC1);
		}

		// Write output to serialized file if it is open
		if (archiveOut_)
		{
			*archiveOut_ << frame_ << depthMat_;
			const int frameSplitCount = 10 ;
			if ((frameNumber_ > 0) && ((frameNumber_ % frameSplitCount) == 0))
			{
				stringstream ofName;
				ofName << change_extension(outFileName_, "").string() << "_" ;
				ofName << (frameNumber_ / frameSplitCount) << ".zms";
				if (!openSerializeOutput(ofName.str().c_str()))
					cerr << "Could not open " << ofName.str() << " for serialized output" << endl;
			}
		}

		while (frame_.rows > 700)
		{
			pyrDown(frame_, frame_);
			pyrDown(depthMat_, depthMat_);
			pyrDown(normDepthMat_, normDepthMat_);
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
	// If we're using an input file we can calculate this.
	if (archiveIn_ && serializeFrameSize_)
		return (((int)serializeIn_->tellg() - serializeFrameStart_) / serializeFrameSize_);

	// Luckily getSVONumberOfFrames() returns -1 if we're
	// capturing from a camera, which is also what the rest
	// of our code expects in that case
	if (zed_)
		return zed_->getSVONumberOfFrames();
		
	// If using a video, there's no way to tell
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
	if (archiveIn_ && serializeFrameSize_)
	{
		serializeIn_->seekg(serializeFrameStart_ + frameNumber_ * serializeFrameSize_);
		frameNumber_ = frameNumber;
	}
	else if (zed_)  
	{
		if (zed_->setSVOPosition(frameNumber))
			frameNumber_ = frameNumber;
	}
}


float ZedIn::getDepth(int x, int y) 
{
	const float* ptr_image_num = (const float*) ((int8_t*)depthMat_.data + y * depthMat_.step);
	return ptr_image_num[x];
}


bool ZedIn::getDepthMat(Mat &depthMat) const
{
	depthMat_.copyTo(depthMat); //not normalized depth
	return true;
}


bool ZedIn::getNormDepthMat(Mat &normDepthMat) const
{
	normDepthMat_.copyTo(normDepthMat); 
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


CameraParams ZedIn::getCameraParams(bool left) const
{
	sl::zed::CamParameters zedp;
	if (zed_)
	{
		if(left)
			zedp = zed_->getParameters()->LeftCam;
		else
			zedp = zed_->getParameters()->RightCam;
	}
	else
	{
		// Take a guess based on acutal values from one of our cameras
		if (width_ == 640)
		{
			zedp.fx = 705.768;
			zedp.fy = 705.768;
			zedp.cx = 326.848;
			zedp.cy = 240.039;
		}
		else if (width_ == 1280)
		{
			zedp.fx = 686.07;
			zedp.fy = 686.07;
			zedp.cx = 662.955;
			zedp.cy = 361.614;
		}
		else if ((width_ == 1920) || (width_ == 960)) // 1920 downscaled
		{
			zedp.fx = 1401.88;
			zedp.fy = 1401.88;
			zedp.cx = 977.193 / (1920 / width_); // Is this correct - downsized
			zedp.cy = 540.036 / (1920 / width_); // image needs downsized cx?
		}
		else if ((width_ == 2208) || (width_ == 1104)) // 2208 downscaled
		{
			zedp.fx = 1385.4;
			zedp.fy = 1385.4;
			zedp.cx = 1124.74 / (2208 / width_);
			zedp.cy = 1124.74 / (2208 / width_);
		}
	}
	CameraParams params;
	if (width_ == 640)
		params.fov = Point2f(51.3 * M_PI / 180, 51.3 / 480. * 640. * M_PI / 180);
	else
		params.fov = Point2f(105 * M_PI / 180, 105 / 720. * 1280. * M_PI / 180); // Guessing all 16:9 resolutions are the same
	params.fx = zedp.fx;
	params.fy = zedp.fy;
	params.cx = zedp.cx;
	params.cy = zedp.cy;
	return params;
}


void zedBrightnessCallback(int value, void *data)
{
    ZedIn *zedPtr = (ZedIn *)data;
	zedPtr->brightness_ = value;
	if (zedPtr->zed_)
	{
		zedPtr->zed_->setCameraSettingsValue(sl::zed::ZED_BRIGHTNESS, value - 1, value == 0);
	}
}


void zedContrastCallback(int value, void *data)
{
    ZedIn *zedPtr = (ZedIn *)data;
	zedPtr->contrast_ = value;
	if (zedPtr->zed_)
	{
		zedPtr->zed_->setCameraSettingsValue(sl::zed::ZED_CONTRAST, value - 1, value == 0);
	}
}


void zedHueCallback(int value, void *data)
{
    ZedIn *zedPtr = (ZedIn *)data;
	zedPtr->hue_ = value;
	if (zedPtr->zed_)
	{
		zedPtr->zed_->setCameraSettingsValue(sl::zed::ZED_HUE, value - 1, value == 0);
	}
}


void zedSaturationCallback(int value, void *data)
{
    ZedIn *zedPtr = (ZedIn *)data;
	zedPtr->saturation_ = value;
	if (zedPtr->zed_)
	{
		zedPtr->zed_->setCameraSettingsValue(sl::zed::ZED_SATURATION, value - 1, value == 0);
	}
}


void zedGainCallback(int value, void *data)
{
    ZedIn *zedPtr = (ZedIn *)data;
	zedPtr->gain_ = value;
	if (zedPtr->zed_)
	{
		zedPtr->zed_->setCameraSettingsValue(sl::zed::ZED_GAIN, value - 1, value == 0);
	}
}


void zedWhiteBalanceCallback(int value, void *data)
{
    ZedIn *zedPtr = (ZedIn *)data;
	zedPtr->whiteBalance_ = value;
	if (zedPtr->zed_)
	{
		zedPtr->zed_->setCameraSettingsValue(sl::zed::ZED_WHITEBALANCE, value - 1, value == 0);
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
