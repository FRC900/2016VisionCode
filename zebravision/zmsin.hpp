#pragma once

//opencv include
#include <opencv2/core/core.hpp>
#include "zedin.hpp"

#include <boost/thread.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <portable_binary_iarchive.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>

class ZMSIn : public ZedIn
{
	public:
		ZMSIn(const char *inFileName = NULL, ZvSettings *settings = NULL);
		~ZMSIn();

#ifdef ZED_SUPPORT
		bool isOpened(void) const;
		bool getFrame(cv::Mat &frame, cv::Mat &depth, bool pause = false);

	private:
		void deleteInputPointers(void);
		bool openSerializeInput(const char *filename, bool portable);
		void update(void);

		// Flag used to syncronize between update and get calls
		bool         frameReady_;

		// frame_ is the most recent frame grabbed from 
		// the camera
		// prevGetFrame_ is the last frame returned from
		// getFrame().  If paused, code needs to keep returning
		// this frame rather than getting a new one from frame_
		cv::Mat      frame_;
		cv::Mat      depth_;
		cv::Mat      prevGetFrame_;
		cv::Mat      prevGetDepth_;

		// Mutex used to protect frame_
		// from simultaneous accesses 
		// by multiple threads
		boost::mutex mtx_;
		
		// Condition variable used to signal between
		// update & getFrame - communicates when a 
		// frame is ready to use or needs to be read
		boost::condition_variable condVar_;

		// Thread object to track update() thread
		boost::thread thread_;

		// Hack up a way to save zed data - serialize both
		// BGR frame and depth frame
		std::ifstream *serializeIn_;
		boost::iostreams::filtering_streambuf<boost::iostreams::input> *filtSBIn_;
		boost::archive::binary_iarchive *archiveIn_;
		portable_binary_iarchive *portableArchiveIn_;
#endif
};
