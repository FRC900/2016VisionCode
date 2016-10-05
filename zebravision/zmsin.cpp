// ZMS file is home-brewed serialization format
// which just dumps raw a image and depth Mat data to a file.
// Apply a light bit of compression because
// the files will get out of hand quickly otherwise
// Initial versions of these files were non-portable
// but later versions were changed to be useable
// on both ARM and x86.  Handle loading both types,
// at least for the time being
#include <iostream>
#include "zmsin.hpp"
using namespace std;

// TODO : this should really be usable even if ZED support
// isn't found.
#ifdef ZED_SUPPORT
#include <boost/filesystem.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cvMatSerialize.hpp"
#include "ZvSettings.hpp"

using namespace cv;
using namespace boost::filesystem;

ZMSIn::ZMSIn(const char *inFileName, ZvSettings *settings) :
	ZedIn(settings),
	frameReady_(false),
	serializeIn_(NULL),
	filtSBIn_(NULL),
	archiveIn_(NULL),
	portableArchiveIn_(NULL)
{
	width_ = 0;
	height_ = 0;
	// Grab the first frame to figure out image size
	cerr << "Loading " << inFileName << " for reading" << endl;
	bool loaded = false;
	if (openSerializeInput(inFileName, true) ||
		openSerializeInput(inFileName, false))
	{
		loaded = true;
		try
		{
			if (archiveIn_)
				*archiveIn_ >> frame_ >> depth_;
			else
				*portableArchiveIn_ >> frame_ >> depth_;
		}
		catch (const std::exception &e)
		{
			loaded = false;
		}
	}

	if (loaded)
	{
		width_  = frame_.cols;
		height_ = frame_.rows;
		initCameraParams(true);

		// Reopen the file so callers can get the first frame
		if (!openSerializeInput(inFileName, archiveIn_ == NULL))
			cerr << "Zed init : Could not reopen " << inFileName << " for reading" << endl;
		else
			thread_ = boost::thread(&ZMSIn::update, this);
	}
	else
	{
		cerr << "Zed init : Could not open " << inFileName << " for reading" << endl;
		deleteInputPointers();
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
bool ZMSIn::openSerializeInput(const char *inFileName, bool portable)
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
void ZMSIn::deleteInputPointers(void)
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


ZMSIn::~ZMSIn()
{
	thread_.interrupt();
	thread_.join();
	deleteInputPointers();
}


bool ZMSIn::isOpened(void) const
{
	return archiveIn_ || portableArchiveIn_;
}


// Read the next frame from the input file.  Store the
// read frame in frame_ & depth_.
// The code is designed not to skip any input frames,
// so if the data stored in frame_&depth_ hasn't been read
// in getFrame yet, update() will loop until it has
// before overwriting it.
void ZMSIn::update(void)
{
	if (!archiveIn_ && !portableArchiveIn_)
		return;

	while (1)
	{
		// If the frame read from the last update()
		// call hasn't been used yet, loop here
		// until it has been. This will prevent
		// the code from reading multiple frames
		// in the time it takes to process one and
		// skipping some video in the process
		boost::mutex::scoped_lock guard(mtx_);
		while (frameReady_)
			condVar_.wait(guard);

		// Ugly try-catch to detect EOF
		try
		{
			if (archiveIn_)
				*archiveIn_ >> frame_ >> depth_;
			else
				*portableArchiveIn_ >> frame_ >> depth_;
		}
		catch (const std::exception &e)
		{
			// EOF reached.  Signal this by sending
			// an empty frame to getFrame.
			frame_ = cv::Mat();
			frameReady_ = true;
			condVar_.notify_all();
			break;
		}

		setTimeStamp(); // TODO : maybe store & read this from the ZMS file instead - this will break the format, though
		incFrameNumber();

		while (frame_.rows > 700)
		{
			pyrDown(frame_, frame_);
			pyrDown(depth_, depth_);
		}

		frameReady_ = true;
		condVar_.notify_all();
	}
}


bool ZMSIn::getFrame(cv::Mat &frame, cv::Mat &depth, bool pause)
{
	if (!archiveIn_ && !portableArchiveIn_)
		return false;

	// If reading from a file and not paused, grab
	// the next frame.
	if (!pause)
	{
		// Wait until a valid frame is in frame_
		boost::mutex::scoped_lock guard(mtx_);
		while (!frameReady_)
			condVar_.wait(guard);
		if (frame_.empty())
			return false;
		frame_.copyTo(prevGetFrame_);
		depth_.copyTo(prevGetDepth_);
		lockTimeStamp();
		lockFrameNumber();

		frameReady_ = false;
		condVar_.notify_all();
	}

	if (prevGetFrame_.empty())
		return false;

	prevGetFrame_.copyTo(frame);
	prevGetDepth_.copyTo(depth);

	return true;
}


#else

ZMSIn::ZMSIn(const char *inFileName, ZvSettings *settings) :
	ZedIn(settings)
{
	(void)inFileName;
}


ZMSIn::~ZMSIn()
{
}

#endif
