// Class to handle input from generic camera sources
// Uses OpenCV's VideoCapture interface to control some
// basic settings but there's not too much interesting
// to do with contrast/brightness/etc.
// Code runs an update thread in the background which 
// constantly polls the camera.  Calls to getFrame()
// return the most recent data (unless pause is set - then
// the last frame called with pause==false is returned)
#pragma once

#include <boost/thread.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "mediain.hpp"

class ZvSettings;

class CameraIn : public MediaIn
{
	public:
		CameraIn(int stream = -1, ZvSettings *settings = NULL);
		~CameraIn();

		bool isOpened(void) const;
		bool getFrame(cv::Mat &frame, cv::Mat &depth, bool pause = false);

	private:
		double           fps_;

		// Input is buffered several times
		// frame_ is the most recent frame grabbed from 
		// the camera
		// pausedFrame_ is the most recent frame returned
		// from a call to getFrame. If video is paused, this
		// frame is returned multiple times until the
		// GUI is unpaused
		cv::Mat           frame_;
		cv::Mat           pausedFrame_;

		// Mutex used to protect frame_
		// from simultaneous accesses 
		// by multiple threads
		boost::mutex      mtx_;

		// Thread dedicated to update() loop
		boost::thread thread_;

		// Flag and condition variable to indicate
		// update() has grabbed at least 1 frame
		boost::condition_variable condVar_;
		bool updateStarted_;

		cv::VideoCapture cap_;

		void update(void);
		std::string getClassName() const { return "CameraIn"; } 
		bool loadSettings(void);
		bool saveSettings(void) const;
};
