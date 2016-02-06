#ifndef MEDIAIN_HPP__
#define MEDIAIN_HPP__

#include <opencv2/core/core.hpp>

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
      virtual int    frameCounter(void) const;
      virtual void   frameCounter(int framecount);

	  // Get depth info for current frame
	  virtual bool   getDepthMat(cv::Mat &depthMat);
      virtual double getDepth(int x, int y);
};
#endif

