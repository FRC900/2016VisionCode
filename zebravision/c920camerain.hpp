// Similar to camerain, but exposes all of the camera controls
// specific to the C920 camera
#pragma once
#include <opencv2/core/core.hpp>

#include "mediain.hpp"

#ifdef __linux__
#include <boost/thread.hpp>
#include "C920Camera.h"
#endif

class ZvSettings;

// Code specific for C920 camera. We have lots of
// extra controls avaiable for this, so use it if
// possible
class C920CameraIn : public MediaIn
{
	public:
		C920CameraIn(int stream = -1, bool gui = false, ZvSettings *settings = NULL);
		~C920CameraIn();
		
#ifdef __linux__  // Special C920 support only works under linux
		bool isOpened(void) const;
		bool getFrame(cv::Mat &frame, cv::Mat &depth, bool pause = false);

		CameraParams getCameraParams(bool left) const;

	private:
		void update(void);

		bool initCamera(bool gui);
		bool loadSettings(void);
		bool saveSettings(void) const;
		std::string getClassName() const { return "C920CameraIn"; } 

		// Mark these as friends so they can access private class data
		friend void brightnessCallback(int value, void *data);
		friend void contrastCallback(int value, void *data);
		friend void saturationCallback(int value, void *data);
		friend void sharpnessCallback(int value, void *data);
		friend void gainCallback(int value, void *data);
		friend void autoExposureCallback(int value, void *data);
		friend void backlightCompensationCallback(int value, void *data);
		friend void whiteBalanceTemperatureCallback(int value, void *data);
		friend void focusCallback(int value, void *data);

		// The camera object itself
		v4l2::C920Camera  camera_;

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

		// Various camera settings
		int               brightness_;
		int               contrast_;
		int               saturation_;
		int               sharpness_;
		int               gain_;
		int               focus_;
		int               autoExposure_;
		int               backlightCompensation_;
		int               whiteBalanceTemperature_;
		v4l2::CaptureSize captureSize_;
		v4l2::CaptureFPS  captureFPS_;
#endif
};
