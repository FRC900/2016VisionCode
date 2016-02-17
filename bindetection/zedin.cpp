#include <iostream>
#include "zedin.hpp"
using namespace std;

#ifdef ZED_SUPPORT
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>

using namespace boost::filesystem;

#include "cvMatSerialize.hpp"

using namespace cv;

ZedIn::ZedIn(const char *inFileName, const char *outFileName) :
	zed_(NULL),
	serializeIn_(NULL), 
	archiveIn_(NULL),
	serializeOut_(NULL),
	archiveOut_(NULL)
{
	if (inFileName)
	{
		// Might be svo, might be zms
		string fnExt = path(inFileName).extension().string();
		if ((fnExt == "svo") || (fnExt == "SVO"))
			zed_ = new sl::zed::Camera(inFileName);
		else if ((fnExt == "zms") || (fnExt == "ZMS"))
		{
			// ZMS file is home-brewed serialization format
			// which just dumps raw Mat data to a file.  
			serializeIn_ = new ifstream(inFileName, ios::in | ios::binary);
			if (serializeIn_ && serializeIn_->is_open())
			{
				filtSBIn_.push(boost::iostreams::zlib_compressor(boost::iostreams::zlib::best_speed));
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
	else
		zed_ = new sl::zed::Camera(sl::zed::VGA, 30);

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
			deletePointers();
		}
		else
		{
			width_     = zed_->getImageSize().width;
			height_    = zed_->getImageSize().height;
			while (height_ > 800)
			{
				width_ /= 2;
				height_ /= 2;
			}
		}
	}
	else if (serializeIn_ && serializeIn_->is_open())
	{
		// Zed == NULL and serializeStream_ means reading from 
		// a serialized file. Grab height_ and width_
		*archiveIn_ >> frame_;
		width_  = frame_.cols;
		height_ = frame_.rows;
		while (height_ > 800)
		{
			width_ /= 2;
			height_ /= 2;
		}
		// Jump back to start of file
		serializeIn_->clear();
		serializeIn_->seekg(0);
	}
	frameCounter_ = 0;
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
	deletePointers();
	if (zed_)
		delete zed_;
}


bool ZedIn::getNextFrame(Mat &frame, bool left, bool pause)
{
	if ((zed_ == NULL) && !(serializeIn_ && serializeIn_->is_open()))
		return false;
	if (pause == false)
	{
		if (zed_)
		{
			zed_->grab(sl::zed::RAW);

			slMat2cvMat(zed_->retrieveImage(left ? sl::zed::SIDE::LEFT : sl::zed::SIDE::RIGHT)).copyTo(frameRGBA_);
			cvtColor(frameRGBA_, frame_, CV_RGBA2RGB);

			slMat2cvMat(zed_->retrieveMeasure(sl::zed::MEASURE::DEPTH)).copyTo(depthMat_); //not normalized depth
			if (serializeOut_ && serializeOut_->is_open())
				*archiveOut_ << frame_ << depthMat_;
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

		while (frame_.rows > 800)
		{
			pyrDown(frame_, frame_);
			pyrDown(depthMat_, depthMat_);
		}
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
