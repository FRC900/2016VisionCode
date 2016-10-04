#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#include "videoin.hpp"
#include "ZvSettings.hpp"

using namespace cv;

VideoIn::VideoIn(const char *inpath, ZvSettings *settings) :
	MediaIn(settings),
	cap_(inpath),
	frameReady_(false), // signal update() to load a new frame immediately
	thread_(boost::bind(&VideoIn::update, this))
{
	if (cap_.isOpened())
	{
		width_  = cap_.get(CV_CAP_PROP_FRAME_WIDTH);
		height_ = cap_.get(CV_CAP_PROP_FRAME_HEIGHT);

		// getNextFrame scales down large inputs
		// make width and height match adjusted frame size
		while (height_ > 700)
		{
			width_ /= 2;
			height_ /= 2;
		}
		frames_ = cap_.get(CV_CAP_PROP_FRAME_COUNT);
	}
	else
		std::cerr << "Could not open input video "<< inpath << std::endl;
}


VideoIn::~VideoIn()
{
	thread_.interrupt();
	thread_.join();
}


bool VideoIn::isOpened(void) const
{
	return cap_.isOpened();
}


// Read the next frame from the input file.  Store the
// read frame in frame_.
// The code is designed not to skip any input frames,
// so if the data stored in frame_ hasn't been read
// in getFrame yet, update() will loop until it has
// before overwriting it.
void VideoIn::update(void)
{
	// Loop until an empty frame is read - 
	// this should identify EOF
	do
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

		cap_ >> frame_;
		setTimeStamp();
		incFrameNumber();
		while (frame_.rows > 700)
			pyrDown(frame_, frame_);

		// Let getFrame know that a frame is ready
		// to be read / processed
		frameReady_ = true;
		condVar_.notify_all();
	}
	while (!frame_.empty());
}


bool VideoIn::getFrame(Mat &frame, Mat &depth, bool pause)
{
	if (!cap_.isOpened())
		return false;

	// If not paused, copy the next frame from
	// frame_. This is the Mat holding the next
	// frame read from the video that update()
	// fills in a separate thread
	if (!pause)
	{
		// Wait until a valid frame is in frame_
		boost::mutex::scoped_lock guard(mtx_);
		while (!frameReady_)
			condVar_.wait(guard);
		if (frame_.empty())
			return false;

		frame_.copyTo(prevGetFrame_);
		lockTimeStamp();
		lockFrameNumber();

		// Let update() know that getFrame has copied
		// the current frame out of frame_
		frameReady_ = false;
		condVar_.notify_all();

		// Release the mutex so that update() can
		// start getting the next frame while the 
		// current one is returned and processed
		// in the main thread.
	}
	if (prevGetFrame_.empty())
		return false;
	prevGetFrame_.copyTo(frame);
	depth = Mat();
	return true;
}


int VideoIn::frameCount(void) const
{
	return frames_;
}


void VideoIn::frameNumber(int frameNumber)
{
	if (frameNumber < frames_)
	{
		cap_.set(CV_CAP_PROP_POS_FRAMES, frameNumber);
		setFrameNumber(frameNumber);
		frameReady_ = false; // force update to read this frame
	}
}
