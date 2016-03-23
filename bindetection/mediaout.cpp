#include <iostream>
#include "mediaout.hpp"
#include "frameticker.hpp"

using namespace cv;

// Set up variables to skip frames and split
// between files as set up by the derived class
// Initialize frameReady shared var to false and 
// kick off the writer thread
MediaOut::MediaOut(int frameSkip, int framesPerFile) :
	frameSkip_(max(frameSkip, 1)),
	frameCounter_(0),
	fileCounter_(0),
	framesPerFile_(framesPerFile),
	frameReady_(false),
	thread_(boost::bind(&MediaOut::writeThread, this))
{
}

MediaOut::~MediaOut(void)
{
  	thread_.interrupt();
  	thread_.join();
}

// Save a frame if there have been frameSkip_ frames
// since the last write (frameSkip == 1 writes every frame,
// == 2 every other, and so on).
// Open a new output file if framesPerFile_ frames have been written
// to the current video
bool MediaOut::saveFrame(const Mat &frame, const Mat &depth)
{
	// Open a new video every framesPerFile_ * frameSkip_ frames.
	// Since frames are written every frameSkip frames, this
	// will put framesPerFile_ frames in each output
	// Since we check this first when frameCounter == 0 on
	// the first frame, this will also open the initial output video
	if ((frameCounter_ % (framesPerFile_ * frameSkip_)) == 0)
	{
		// Lock anything which changes the file
		// pointers just in case
		boost::mutex::scoped_lock lock(fileLock_);
		if (!openNext())
			return false;
	}

	// Every frameSkip_ frames, write another frame
	// to the frame_ and depth_ vars. Then set frameReady_
	// to trigger the writer thread to grab them and
	// write them to disk
	if ((frameCounter_++ % frameSkip_) == 0)
	{
		boost::mutex::scoped_lock lock(matLock_);
		frame.copyTo(frame_);
		depth.copyTo(depth_);
		frameReady_ = true;
		frameCond.notify_one();
	}

	// If we made it this far, the write was successful.
	// Note that it might be that nothing was written if
	// this frame was skipped, but that's not an error
	return true;
}

// Dummy member functions - base class shouldn't be called
// directly so these shouldn't be used
bool MediaOut::openNext(void)
{
	return false;
}

bool MediaOut::write(const Mat &frame, const Mat &depth)
{
	(void)frame;
	(void)depth;
	return false;
}

// Separate thread to write video frames to disk
// This runs as quickly as possible, but is also
// designed to drop frames if it gets behind the
// main processing thread.  
void MediaOut::writeThread(void)
{
	Mat frame;
	Mat depth;
	FrameTicker ft;
	while (true)
	{
		// Grab the lock mutex
		// Check that frameReady is set
		//  if it hasn't, that means there's
		//  no new data to write.  In that case,
		//  call wait() to release the mutex and 
		//  try again
		// Once frameReady_ has been set, copy
		// the data out of the member variables
		// into a local var, release the lock, 
		// and do the write with the copies
		// of the Mats.  Using the copied will
		// let saveFrame write to the shared vars
		// while write() works on the old frame.
		// Note that saveFrame intentionally
		// doesn't check to see if frame_ and
		// depth_ are valid before writing to them.
		// This way if the write() call takes too
		// long it is possible for saveFrame to
		// update the frame_ and depth_ vars more 
		// than once before this thread reads them.
		// That will potentially drop frames, but it
		// also lets the main thread run as quick
		// as possible rather than waiting on this thread
		{
			boost::mutex::scoped_lock lock(matLock_);
			while (!frameReady_)
				frameCond.wait(lock);
			frame_.copyTo(frame);
			depth_.copyTo(depth);
			frameReady_ = false;
		}

		// Lock access to the file, just in case
		{
			boost::mutex::scoped_lock lock(fileLock_);
			write(frame, depth);
		}
		ft.mark();
		std::cout << std::setprecision(2) << ft.getFPS() << " Write FPS" << std::endl;

		boost::this_thread::interruption_point();
	}
}
