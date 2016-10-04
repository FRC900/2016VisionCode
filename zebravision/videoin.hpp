// Input class to handle RGB (or grayscale?) video inputs :
// MPG, AVI, MP4, etc.
// Code runs a separate decode thread which tries to buffer
// one frame ahead of the data needed by getFrame
#pragma once

#include <boost/thread.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "mediain.hpp"

class ZvSettings;

class VideoIn : public MediaIn
{
	public:
		VideoIn(const char *inpath, ZvSettings *settings = NULL);
		~VideoIn();

		bool isOpened(void) const;
		bool getFrame(cv::Mat &frame, cv::Mat &depth, bool pause = false);

		int  frameCount(void) const;
		void frameNumber(int frameNumber);

	private:
		cv::VideoCapture cap_;
		int              frames_;

		// Flag used to syncronize between update and get calls
		bool             frameReady_;

		// frame_ is the most recent frame grabbed from 
		// the camera
		// prevGetFrame_ is the last frame returned from
		// getFrame().  If paused, code needs to keep returning
		// this frame rather than getting a new one from frame_
		cv::Mat           frame_;
		cv::Mat           prevGetFrame_;

		// Mutex used to protect frame_
		// from simultaneous accesses 
		// by multiple threads
		boost::mutex      mtx_;
		
		// Condition variable used to signal between
		// update & getFrame - communicates when a 
		// frame is ready to use or needs to be read
		boost::condition_variable condVar_;

		// Thread object to track update() thread
		boost::thread thread_;

		void update(void);
		bool loadSettings(void) { return true; }
		bool saveSettings(void) const { return true; }
		std::string getClassName() const { return "VideoIn"; }
};
