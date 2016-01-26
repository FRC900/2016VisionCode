#ifndef MEDIAIN_HPP__
#define MEDIAIN_HPP__

#include "opencv2/highgui/highgui.hpp"

using namespace cv;

class MediaIn
{
   public:
      MediaIn();
      virtual bool   getNextFrame(cv::Mat &frame, bool pause = false) = 0;
      virtual int    width() = 0;
      virtual int    height() = 0;
      virtual int    frameCount(void); 
      virtual int    frameCounter(void);
      virtual void   frameCounter(int framecount);
      virtual double getDepth(int x, int y);
};
#endif

