#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>
#include <boost/iostreams/filter/zlib.hpp>

#include "zmsout.hpp"
#include "cvMatSerialize.hpp"

using namespace std;
using namespace cv;
using namespace boost::filesystem;

// Save the raw camera stream to disk.  This uses a home-brew
// method to serialize image and depth data to disk rather than
// relying on Stereolab's SVO format.
ZMSOut::ZMSOut(const char *outFile, int frameSkip) :
	fileName_(outFile),
	serializeOut_(NULL),
	filtSBOut_(NULL),
	archiveOut_(NULL) ,
	frameSkip_(max(frameSkip,1)),
	frameCounter_(0),
	fileCounter_(0)
{
	if (outFile)
		openNext();
}

ZMSOut::~ZMSOut()
{
	deleteOutputPointers();
}

bool ZMSOut::saveFrame(const Mat &frame, const Mat &depth)
{
	// Write output to serialized file if it is open
	// if we've skipped enough frames since the last write 
	// (which could be every frame if frameSkip == 0 or 1
	if (archiveOut_ && ((frameCounter_++ % frameSkip_) == 0))
	{
		*archiveOut_ << frame << depth;
		const int frameSplitCount = 300;
		if ((frameCounter_ > 1) && (((frameCounter_ - 1) % frameSplitCount) == 0))
			return openNext();
	}
	return true;
}

// Output needs 3 things. First is a standard ofstream to write to
// Next is an (optional) filtered stream buffer. This is used to
// compress on the fly - uncompressed files take up way too
// much space. Last item is the actual boost binary archive template
// If all three are opened, return true. If not, delete and set to
// NULL all pointers related to serialized Output
bool ZMSOut::openSerializeOutput(const char *fileName)
{
	deleteOutputPointers();
	serializeOut_ = new ofstream(fileName, ios::out | ios::binary);
	if (!serializeOut_ || !serializeOut_->is_open())
	{
		cerr << "Could not open ofstream(" << fileName << ")" << endl;
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

bool ZMSOut::openNext(void)
{
	stringstream ofName;
	ofName << change_extension(fileName_, "").string() << "_" ;
	ofName << fileCounter_++ << ".zms";
	if (!openSerializeOutput(ofName.str().c_str()))
	{
		cerr << "ZMSOut : could not open output file " << ofName.str() << endl;
		return false;
	}
	return true;
}


// Helper to easily delete and NULL out output file pointers
void ZMSOut::deleteOutputPointers(void)
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
