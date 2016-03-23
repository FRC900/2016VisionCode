#pragma once
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include "mediaout.hpp"

// Hack up a way to save zed data - serialize both
// BGR frame and depth frame
class ZMSOut : public MediaOut
{
	public:
		ZMSOut(const char *outFile, int frameSkip = 0);
		~ZMSOut();
		bool saveFrame(const cv::Mat &frame, const cv::Mat &depth);

	private :
		bool openNext(void);
		void deleteOutputPointers(void);
		bool openSerializeOutput(const char *filename);
		cv::Mat     frame_;
		cv::Mat     depth_;
		std::string fileName_;

		std::ofstream *serializeOut_;
		boost::iostreams::filtering_streambuf<boost::iostreams::output> *filtSBOut_;
		boost::archive::binary_oarchive *archiveOut_;

		// Skip output frames if requested.  Skip is how many to 
		// skip before writing next output frame, FrameCounter is how
		// many total frames seen.
		// Counter is used to split the output into multiple shorter
		// outputs
		int frameSkip_;
		int frameCounter_;
		int fileCounter_;
};
