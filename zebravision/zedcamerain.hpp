// Class to handle video input from ZED camera
// Similar to other camerain classes, except that depth
// data is returned from getFrame.
// Note that this class is derived from ZedIn rather than 
// MediaIn like many of the other *in classes. This lets the
// code use a common call to grab the camera parameters.
// TODO : test to see that this is correct for SVO files.
// If not, break the code up since calls from ZMS in use fakes
// of those values
#pragma once

#include <boost/thread.hpp>
#include <opencv2/core/core.hpp>
#include "zedin.hpp"

#ifdef ZED_SUPPORT
#endif

class ZvSettings;

class ZedCameraIn : public ZedIn
{
	public:
		ZedCameraIn(bool gui = false, ZvSettings *settings = NULL);
		~ZedCameraIn();

#ifdef ZED_SUPPORT
		bool isOpened(void) const;
		bool getFrame(cv::Mat &frame, cv::Mat &depth, bool pause = false);

	private:
		void update(void);
		bool loadSettings(void);
		bool saveSettings(void) const;
		std::string getClassName() const { return "ZedCameraIn"; }

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
		cv::Mat      pausedFrame_;
		cv::Mat      pausedDepth_;

		// Mutex used to protect frame_
		// from simultaneous accesses 
		// by multiple threads
		boost::mutex mtx_;

		// Used to track update() thread
		boost::thread thread_;

		// Flag and condition variable to indicate
		// update() has grabbed at least 1 frame
		boost::condition_variable condVar_;
		bool updateStarted_;


		int brightness_;
		int contrast_;
		int hue_;
		int saturation_;
		int gain_;

		// Mark these as friends so they can access private class data
		friend void zedBrightnessCallback(int value, void *data);
		friend void zedContrastCallback(int value, void *data);
		friend void zedHueCallback(int value, void *data);
		friend void zedSaturationCallback(int value, void *data);
		friend void zedGainCallback(int value, void *data);
#endif
};
