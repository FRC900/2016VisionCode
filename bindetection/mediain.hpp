#ifndef MEDIAIN_HPP__
#define MEDIAIN_HPP__

#include <opencv2/core/core.hpp>

#include <zed/Camera.hpp>

using namespace cv;

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
	  virtual sl::zed::CamParameters getCameraParams(bool left) const;
	  virtual bool   getDepthMat(cv::Mat &depthMat);
      virtual double getDepth(int x, int y);
};
#endif

