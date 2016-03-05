#ifndef VIDEOIN_HPP__
#define VIDEOIN_HPP__

#include <opencv2/highgui/highgui.hpp>
#include "mediain.hpp"

class VideoIn : public MediaIn
{
   public:
      VideoIn(const char *path);
	  ~VideoIn() {}
      bool getNextFrame(cv::Mat &frame, bool pause = false);
      int width() const;
      int height() const;
      int frameCount(void) const;
      int frameNumber(void) const;
      void frameNumber(int frameNumber);

   private:
      cv::VideoCapture cap_;
      cv::Mat          frame_;
	  int              width_;
	  int              height_;
	  int              frames_;
	  int              frameNumber_;
};
#endif

