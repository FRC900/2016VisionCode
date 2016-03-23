#include "mediaout.hpp"

using namespace cv;

// Set up variables to skip frames and split
// between files as set up by the derived class
MediaOut::MediaOut(int frameSkip, int framesPerFile) :
	frameSkip_(max(frameSkip, 1)),
	frameCounter_(0),
	fileCounter_(0),
	framesPerFile_(framesPerFile)
{
}

MediaOut::~MediaOut(void)
{
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
		if (!openNext())
			return false;;

	// Every frameSkip_ frames, write another frame to
	// the output
	// Return false if an attempt to write a frame failed
	if (((frameCounter_++ % frameSkip_) == 0) && !write(frame, depth))
		return false;

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
