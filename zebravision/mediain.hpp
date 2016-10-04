#ifndef MEDIAIN_HPP__
#define MEDIAIN_HPP__

#include <opencv2/core/core.hpp>
#include <boost/thread.hpp>
#include <tinyxml2.h>

#include "frameticker.hpp"
#include "ZvSettings.hpp"

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
		virtual bool isOpened(void) const;
		virtual bool update(void) = 0;
		virtual bool getFrame(cv::Mat &frame, cv::Mat &depth, bool pause = false) = 0;

		// Image size
		unsigned int width() const;
		unsigned int height() const;

		// How many frames?
		virtual int frameCount(void) const;

		// Get and set current frame number
		virtual void frameNumber(int frameNumber);

		int frameNumber(void) const;
		long long timeStamp(void) const;

		virtual float FPS(void) const;

		// Other functions that really only work from zedin
		virtual CameraParams getCameraParams(bool left) const;

	protected:
		unsigned int width_;
		unsigned int height_;
		cv::Mat frame_;
		boost::mutex mtx_;
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
		int       frameNumber_;
		int       lockedFrameNumber_;
		long long timeStamp_;
		long long lockedTimeStamp_;
		FrameTicker frameTicker;
};
#endif
