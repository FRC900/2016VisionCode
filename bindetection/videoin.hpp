#ifndef VIDEOIN_HPP__
#define VIDEOIN_HPP__

#include <opencv2/highgui/highgui.hpp>
#include "mediain.hpp"

class VideoIn : public MediaIn
{
   public:
      VideoIn(const char *inpath, const char *outpath = NULL);
	  ~VideoIn() {}
      bool update();
      bool getFrame(cv::Mat &frame, cv::Mat &depth);
      bool saveFrame(cv::Mat &frame, cv::Mat &depth);
      int width() const;
      int height() const;
      int frameCount(void) const;
      int semValue(void) { return 1; }
      int frameNumber(void) const;
      void frameNumber(int frameNumber);

   private:
    bool increment;
    cv::VideoCapture cap_;
    cv::VideoWriter  writer_;
	  int              width_;
	  int              height_;
	  int              frames_;
	  int              frameNumber_;
};
#endif
