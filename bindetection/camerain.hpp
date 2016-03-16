#ifndef CAMERAIN_HPP__
#define CAMERAIN_HPP__

#include <opencv2/highgui/highgui.hpp>
#include "mediain.hpp"

class CameraIn : public MediaIn
{
   public:
      CameraIn(const char* outfile,int stream = -1, bool gui = false);
	    ~CameraIn() {}
      bool getFrame(cv::Mat &frame, cv::Mat &depth);
      bool saveFrame(cv::Mat &frame, cv::Mat &depth);
      bool update();
      int width(void) const;
      int height(void) const;
      int frameNumber(void) const;
      bool saveFrame(cv::Mat &frame);

   private:
    int              lockedFrameNumber_;
    int              frameNumber_;
	  int              width_;
	  int              height_;

    cv::VideoCapture cap_;
    cv::VideoWriter writer_;
};
#endif
