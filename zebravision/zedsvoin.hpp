// Use ZED SVO input files. Similar to other video in classes.
// Runs an update thread in the background which tries to
// decode one frame ahead of the frame used in getFrame().

#pragma once

#include <boost/thread.hpp>
#include <opencv2/core/core.hpp>
#include "zedin.hpp"

class ZedSVOIn : public ZedIn
{
	public:
		ZedSVOIn(const char *inFileName = NULL, ZvSettings *settings = NULL);
		~ZedSVOIn();

#ifdef ZED_SUPPORT
		bool isOpened(void) const;
		bool getFrame(cv::Mat &frame, cv::Mat &depth, bool pause = false);

		// How many frames?
		int    frameCount(void) const;

		// Get and set current frame number
		int    frameNumber(void) const;
		void   frameNumber(int frameNumber);

	private:
		void update(void);
		// Input is buffered several times
		// RGB and depth are stored in separate cv::Mat objects
		// frame_/depth_ is the most recent frame grabbed from 
		// the camera
		// pausedFrame_/pausedDepth_ is the most recent frame returned
		// from a call to getFrame. If video is paused, this
		// frame is returned multiple times until the
		// GUI is unpaused
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

		// Used to track update thread
		boost::thread thread_;

		// Flag used to syncronize between update and get calls
		bool             frameReady_;
#endif
};
