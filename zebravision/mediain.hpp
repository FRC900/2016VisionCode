// Base class for all input - cameras, videos, stills
// Has the basics all inputs use - current frame number,
// width& height, and a timestamp the current frame was aquired
// Other methods are declared as pure virtual which forces
// derived classes to implement them with the particulars
// of that input type.
#pragma once

#include <opencv2/core/core.hpp>
#include <tinyxml2.h>

#include "frameticker.hpp"
#include "ZvSettings.hpp"

// Mimic ZEDg camera parameters instead of using them - allows
// for targets without ZED support to still get this info
// for other cameras
class CameraParams
{
	public:
		CameraParams() :
			fov(51.3 * M_PI / 180., 51.3 * 480. / 640. * M_PI / 180.), // Default to zed params?
			fx(0),
			fy(0),
			cx(0),
			cy(0)
		{
			for (size_t i = 0; i < sizeof(disto)/sizeof(disto[0]); i++)
				disto[i] = 0.0;
		}
		cv::Point2f fov;
		float       fx;
		float       fy;
		float       cx;
		float       cy;
		double      disto[5];
};

// Base class for input.  Derived classes are cameras, videos, etc
class MediaIn
{
	public:
		MediaIn(ZvSettings *settings);
		virtual ~MediaIn() {}

		// These should be implemented by each derived class
		virtual bool isOpened(void) const = 0;
		virtual bool update(void) = 0;
		virtual bool getFrame(cv::Mat &frame, cv::Mat &depth, bool pause = false) = 0;

		// Image size
		unsigned int width() const;
		unsigned int height() const;

		// How many frames?
		virtual int frameCount(void) const;

		// Set current frame number
		virtual void frameNumber(int frameNumber);

		// Get frame number and the time that frame was
		// captured. Former only makes sense for video
		// input and the latter for live camera feeds
		int frameNumber(void) const;
		long long timeStamp(void) const;

		// Input FPS for live camera input
		virtual float FPS(void) const;

		// Camera parameters - fov, focal length, etc.
		virtual CameraParams getCameraParams(bool left) const;

	protected:
		// Width and height of input frame
		unsigned int width_;
		unsigned int height_;

		// Saved settings for this input type
		ZvSettings *settings_;

		void setTimeStamp(long long timeStamp = -1);
		void lockTimeStamp(void);
		void setFrameNumber(int frameNumber);
		void incFrameNumber(void);
		void lockFrameNumber(void);
		void FPSmark(void);

		virtual bool loadSettings(void);
		virtual bool saveSettings(void) const;
		virtual std::string getClassName() const { return "MediaIn"; }

	private:
		// Maintain two sets of frame numbers and time stamps.
		// The locked version corresponds to the frame that was
		// current the last time getFrame was called.  The other
		// one is updated for each input frame. Since the update()
		// threads can run at a different speed than getFrame is called,
		// this lets the code maintain the correct values associated 
		// with each - i.e. multiple calls to a paused getFrame() will
		// return the same locked* value even as update() changes the
		// non-locked versions in a separate thread
		int       frameNumber_;
		int       lockedFrameNumber_;
		long long timeStamp_;
		long long lockedTimeStamp_;
		FrameTicker frameTicker;
};
