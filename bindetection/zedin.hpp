#ifndef ZEDCAMERAIN_HPP__
#define ZEDCAMERAIN_HPP__

//opencv include
#include "opencv2/highgui/highgui.hpp"
#include "mediain.hpp"

//zed include
#include <zed/Mat.hpp>
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>

#ifdef _linux
#endif

class ZedIn : public MediaIn
{
   public:
      ZedIn();
      bool getNextFrame(cv::Mat &frame, bool pause = false);

      int width(void);
      int height(void);
      double getDepth(int x, int y);
      
   private:
	sl::zed::Camera* zed;
	sl::zed::Mat imageGPU;
	sl::zed::Mat depthGPU;
	cv::Mat depthCPU;
        sl::zed::Mat depthMat;
 	bool getNextFrame(cv::Mat &frame, bool left, bool pause);
	int width_;
	int height_;
};
#endif

