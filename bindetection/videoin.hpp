#ifndef VIDEOIN_HPP__
#define VIDEOIN_HPP__

#include <opencv2/highgui/highgui.hpp>
#include "mediain.hpp"

class VideoIn : public MediaIn
{
   public:
      VideoIn(const char *path);
      bool getNextFrame(cv::Mat &frame, bool pause = false);
      int width() const;
      int height() const;
      int frameCount(void) const;
      int frameCounter(void) const;
      void frameCounter(int frameCounter);

   private:
      cv::VideoCapture cap_;
      cv::Mat          frame_;
	  int              width_;
	  int              height_;
	  int              frames_;
	  int              frameCounter_;
};
#endif

