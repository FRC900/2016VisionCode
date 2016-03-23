#ifndef MEDIAIN_HPP__
#define MEDIAIN_HPP__

#include <opencv2/core/core.hpp>
#include <boost/thread.hpp>

class CameraParams
{
	public:
		CameraParams() :
			fov(51.3 * M_PI / 180., 51.3 * 480. / 640. * M_PI / 180.), // Default to zed params?
			fx(0),
			fy(0),
			cx(0),
			cy(0)
		{}
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
		MediaIn(void);
		virtual ~MediaIn() {}
		virtual bool getFrame(cv::Mat &frame, cv::Mat &depth) = 0;
		virtual bool update() = 0;

		// Image size
		virtual int    width() const = 0;
		virtual int    height() const = 0;

		// How many frames?
		virtual int    frameCount(void) const;

		//initial semaphore value
		virtual int semValue();

		// Get and set current frame number
		virtual int    frameNumber(void) const;
		virtual void   frameNumber(int frameNumber);

		// Other functions that really only work from zedin
		virtual CameraParams getCameraParams(bool left) const;

	protected:
		cv::Mat _frame;
		boost::mutex _mtx;
};
#endif
