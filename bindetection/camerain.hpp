#ifndef CAMERAIN_HPP__
#define CAMERAIN_HPP__

#include <opencv2/highgui/highgui.hpp>
#include "mediain.hpp"

class CameraIn : public MediaIn
{
   public:
      CameraIn(int stream = -1, bool gui = false);
	  ~CameraIn() {}
      bool getNextFrame(cv::Mat &frame, bool pause = false);

      int width(void) const;
      int height(void) const;
      int frameCounter(void) const;

   private:
      int              frameCounter_;
	  int              width_;
	  int              height_;
      cv::Mat          frame_;
      cv::VideoCapture cap_;
};
#endif
