#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>

#include "aviout.hpp"

using namespace std;
using namespace cv;
using namespace boost::filesystem;

AVIOut::AVIOut(const char *outFile, const Size &size, int frameSkip):
	size_(size),
	writer_(NULL),
	fileName_(outFile),
	frameSkip_(max(frameSkip,1)),
	frameCounter_(0),
	fileCounter_(0)
{
	// open the output video
	if(outFile != NULL) 
		openNext();
}

AVIOut::~AVIOut()
{
	if (writer_)
		delete writer_;
}

bool AVIOut::saveFrame(const Mat &frame, const Mat &depth)
{
	(void)depth;
	if (writer_ && writer_->isOpened() && ((frameCounter_++ % frameSkip_) == 0))
	{
		*writer_ << frame;
		const int frameSplitCount = 300;
		if ((frameCounter_ > 1) && (((frameCounter_ - 1) % frameSplitCount) == 0))
			return openNext();
	}
	return true;
}

bool AVIOut::openNext(void)
{
	if (writer_)
	{
		delete writer_;
		writer_ = NULL;
	}
	stringstream ofName;
	ofName << change_extension(fileName_, "").string() << "_" ;
	ofName << fileCounter_++ << ".avi";
	writer_ = new VideoWriter(ofName.str(), CV_FOURCC('M','J','P','G'), 30, size_, true);
	if(!writer_ || !writer_->isOpened())
	{
		std::cerr << "AVIOut() : Could not open output video " << ofName.str() << std::endl;
		delete writer_;
		writer_ = NULL;
		return false;
	}
	return true;
}
