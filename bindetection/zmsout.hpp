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

	private :
		bool openNext(void);
		void deleteOutputPointers(void);
		bool openSerializeOutput(const char *filename);
		bool write(const cv::Mat &frame, const cv::Mat &depth);

		std::string fileName_;

		std::ofstream *serializeOut_;
		boost::iostreams::filtering_streambuf<boost::iostreams::output> *filtSBOut_;
		boost::archive::binary_oarchive *archiveOut_;
};