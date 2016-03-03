#ifndef MEDIAIN_HPP__
#define MEDIAIN_HPP__

#include <opencv2/core/core.hpp>

using namespace cv;

class CameraParams
{
	public:
		CameraParams() :
			fov(51.3, 51.3 * 480. / 640.), // Default to zed params?
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
      MediaIn();
	  virtual ~MediaIn() {}
      virtual bool   getNextFrame(cv::Mat &frame, bool pause = false) = 0;

	  // Image size
      virtual int    width() const = 0;
      virtual int    height() const = 0;

	  // How many frames?
      virtual int    frameCount(void) const; 

	  // Get and set current frame number
      virtual int    frameNumber(void) const;
      virtual void   frameNumber(int frameNumber);

	  // Get depth info for current frame
	  virtual CameraParams getCameraParams(bool left) const;
	  virtual bool  getDepthMat(cv::Mat &depthMat) const;
	  virtual bool  getNormDepthMat(cv::Mat &normDepthMat) const;
      virtual float getDepth(int x, int y);
};
#endif

