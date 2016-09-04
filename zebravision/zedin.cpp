#include <iostream>
#include "zedin.hpp"
using namespace std;

#ifdef ZED_SUPPORT
#include <boost/filesystem.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cvMatSerialize.hpp"
#include "ZvSettings.hpp"

using namespace cv;
using namespace boost::filesystem;

void zedBrightnessCallback(int value, void *data);
void zedContrastCallback(int value, void *data);
void zedHueCallback(int value, void *data);
void zedSaturationCallback(int value, void *data);
void zedGainCallback(int value, void *data);
void zedWhiteBalanceCallback(int value, void *data);

ZedIn::ZedIn(const char *inFileName, bool gui, ZvSettings *settings) :
	MediaIn(settings),
	zed_(NULL),
	width_(0),
	height_(0),
	brightness_(2),
	contrast_(6),
	hue_(7),
	saturation_(4),
	gain_(1),
	whiteBalance_(3101),
	serializeIn_(NULL),
	filtSBIn_(NULL),
	archiveIn_(NULL),
	portableArchiveIn_(NULL)
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
			// Initial versions of these files were non-portable
			// but later versions were changed to be useable
			// on both ARM and x86.  Handle loading both types,
			// at least for the time being
			cerr << "Loading " << inFileName << " for reading" << endl;
			bool loaded = false;
			if (openSerializeInput(inFileName, true))
			{
				loaded = true;
				try
				{
					*portableArchiveIn_ >> frame_ >> depthMat_;
				}
				catch (const std::exception &e)
				{
					loaded = false;
				}
			}
			else if (openSerializeInput(inFileName, false))
			{
				loaded = true;
				try
				{
					*archiveIn_ >> frame_ >> depthMat_;
				}
				catch (const std::exception &e)
				{
					loaded = false;
				}
			}
			else
			{
				cerr << "Zed init : Could not open " << inFileName << " for reading" << endl;
			}
			if (loaded)
			{
				width_  = frame_.cols;
				height_ = frame_.rows;

				// Reopen the file so callers can get the first frame
				if (!openSerializeInput(inFileName, archiveIn_ == NULL))
					cerr << "Zed init : Could not reopen " << inFileName << " for reading" << endl;
			}
			else
			{
				deleteInputPointers();
			}
		}
	}
	else // Open an actual camera for input
		zed_ = new sl::zed::Camera(sl::zed::HD720, 30);

	if (zed_)
	{
		// init computation mode of the zed
		sl::zed::ERRCODE err = zed_->init(sl::zed::MODE::QUALITY, -1, true);
		// Quit if an error occurred
		if (err != sl::zed::SUCCESS)
		{
			cout << sl::zed::errcode2str(err) << endl;
			delete zed_;
			zed_ = NULL;
		}
		else
		{
			// Make sure there's at least one good frame read
			// before kicking off the main capture thread
			if (!inFileName && !update())
			{
				cerr << "********** TOO MANY BAD FRAMES, ABORTING" << endl;
				delete zed_;
				zed_ = NULL;
				return;
			}
			width_  = zed_->getImageSize().width;
			height_ = zed_->getImageSize().height;

			if (!loadSettings())
				cerr << "Failed to load ZedIn settings from XML" << endl;

			zedBrightnessCallback(brightness_, this);
			zedContrastCallback(contrast_, this);
			zedHueCallback(hue_, this);
			zedSaturationCallback(saturation_, this);
			zedGainCallback(gain_, this);
			zedWhiteBalanceCallback(whiteBalance_, this);

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

	while (height_ > 700)
	{
		width_  /= 2;
		height_ /= 2;
	}
}

bool ZedIn::loadSettings(void)
{
	if (settings_) {
		settings_->getInt(getClassName(), "brightness",   brightness_);
		settings_->getInt(getClassName(), "contrast",     contrast_);
		settings_->getInt(getClassName(), "hue",          hue_);
		settings_->getInt(getClassName(), "saturation",   saturation_);
		settings_->getInt(getClassName(), "gain",         gain_);
		settings_->getInt(getClassName(), "whiteBalance", whiteBalance_);
		return true;
	}
	return false;
}

bool ZedIn::saveSettings(void) const
{
	if (settings_) {
		settings_->setInt(getClassName(), "brightness",   brightness_);
		settings_->setInt(getClassName(), "contrast",     contrast_);
		settings_->setInt(getClassName(), "hue",          hue_);
		settings_->setInt(getClassName(), "saturation",   saturation_);
		settings_->setInt(getClassName(), "gain",         gain_);
		settings_->setInt(getClassName(), "whiteBalance", whiteBalance_);
		settings_->save();
		return true;
	}
	return false;
}

// Input needs 3 things. First is a standard ifstream to read from
// Next is an (optional) filtered stream buffer. This is used to
// uncompress on the fly - uncompressed files take up way too
// much space. Last item is the actual boost binary archive template
// If all three are opened, return true. If not, delete and set to
// NULL all pointers related to serialized Input
bool ZedIn::openSerializeInput(const char *inFileName, bool portable)
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
	if (portable)
	{
		try
		{
			portableArchiveIn_ = new portable_binary_iarchive(*filtSBIn_);
		}
		catch (std::exception &e)
		{
			portableArchiveIn_ = NULL;
		}
		if (!portableArchiveIn_)
		{
			cerr << "Could not create new portable_binary_iarchive" << endl;
			deleteInputPointers();
			return false;
		}
	}
	else
	{
		try
		{
			archiveIn_ = new boost::archive::binary_iarchive(*filtSBIn_);
		}
		catch (std::exception &e)
		{
			portableArchiveIn_ = NULL;
		}
		if (!archiveIn_)
		{
			cerr << "Could not create new binary_iarchive" << endl;
			deleteInputPointers();
			return false;
		}
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
	if (portableArchiveIn_)
	{
		delete portableArchiveIn_;
		portableArchiveIn_ = NULL;
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


ZedIn::~ZedIn()
{
	if (!saveSettings())
		cerr << "Failed to save ZedIn settings to XML" << endl;
	deleteInputPointers();
	if (zed_)
		delete zed_;
}


bool ZedIn::isOpened(void) const
{
	return zed_ || archiveIn_ || portableArchiveIn_;
}


bool ZedIn::update(bool left)
{
	// Read from either the zed camera or from
	// a previously-serialized ZMS file
	if (zed_)
	{
		int badReadCounter = 0;
		while (zed_->grab(sl::zed::SENSING_MODE::RAW))
		{
			cerr << "********---------GRAB RETURNED FALSE " << endl;
			usleep(33333);
			// If there is an existing frame and the
			// grab fails, just return. This will
			// cause the code to use the last good frame
			if (!frame_.empty())
				return true;
			// Otherwise try to grab a bunch of times before
			// bailing out and failing
			if (++badReadCounter == 100)
				return false;
		}

		slDepth_ = zed_->retrieveMeasure(sl::zed::MEASURE::DEPTH);
		slFrame_ = zed_->retrieveImage(left ? sl::zed::SIDE::LEFT : sl::zed::SIDE::RIGHT);
		boost::lock_guard<boost::mutex> guard(mtx_);
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
	else if (archiveIn_ || portableArchiveIn_)
		usleep (50000);
	else
		return false;

	return true;
}


bool ZedIn::getFrame(cv::Mat &frame, cv::Mat &depth, bool pause)
{
	if (!zed_ && !archiveIn_ && !portableArchiveIn_)
		return false;

	// If reading from a file and not paused, grab
	// the next frame.
	if (!pause && (archiveIn_ || portableArchiveIn_))
	{
		// Ugly try-catch to detect EOF
		try
		{
			if (archiveIn_)
				*archiveIn_ >> frame_ >> depthMat_;
			else
				*portableArchiveIn_ >> frame_ >> depthMat_;
		}
		catch (const std::exception &e)
		{
			return false;
		}
		setTimeStamp(); // TODO : read this from the ZMS file instead - this will break the format, though
		incFrameNumber();

		while (frame_.rows > 700)
		{
			pyrDown(frame_, frame_);
			pyrDown(depthMat_, depthMat_);
		}
	}

	// If video is paused, this will just re-use the
	// previous frame.  If camera input, this
	// will return the last frame read in update -
	// this can also be paused by the calling code
	// if desired
	boost::lock_guard<boost::mutex> guard(mtx_);
	lockTimeStamp();
	lockFrameNumber();

	frame_.copyTo(frame);
	depthMat_.copyTo(depth);
	return true;
}


bool ZedIn::update(void)
{
	return update(true);
}


int ZedIn::frameCount(void) const
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
void ZedIn::frameNumber(int frameNumber)
{
	if (zed_ && zed_->setSVOPosition(frameNumber))
		setFrameNumber(frameNumber);
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
		if (height_ == 480)
		{
			zedp.fx = 705.768;
			zedp.fy = 705.768;
			zedp.cx = 326.848;
			zedp.cy = 240.039;
		}
		else if ((width_ == 1280) || (width_ == 640)) // 720P normal or pyrDown 1x
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
		else
		{
			// This should never happen
			zedp.fx = 0;
			zedp.fy = 0;
			zedp.cx = 0;
			zedp.cy = 0;
		}
	}
	float hFovDegrees;
	if (height_ == 480) // can't work based on width, since 1/2 of 720P is 640, as is 640x480
		hFovDegrees = 51.3;
	else
		hFovDegrees = 105.; // hope all the HD & 2k res are the same
	float hFovRadians = hFovDegrees * M_PI / 180.0;

	CameraParams params;
	params.fov = Point2f(hFovRadians, hFovRadians * (float)height_ / (float)width_);
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


ZedIn::ZedIn(const char *inFileName, bool gui, ZvSettings *settings) :
	MediaIn(settings)
{
	(void)inFileName;
	(void)gui;
	(void)settings;
	cerr << "Zed support not compiled in" << endl;
}


#endif
