#ifndef CAMERAIN_HPP__
#define CAMERAIN_HPP__

#include <opencv2/highgui/highgui.hpp>
#include "mediain.hpp"

class CameraIn : public MediaIn
{
   public:
      CameraIn(int stream = -1, bool gui = false);
	  ~CameraIn() {}
      bool getFrame(cv::Mat &frame);
      bool update();
      int width(void) const;
      int height(void) const;
      int frameNumber(void) const;

   private:
      int              frameNumber_;
	  int              width_;
	  int              height_;
      cv::VideoCapture cap_;
};
#endif
