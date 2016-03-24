#ifndef C920CAMERAIN_HPP__
#define C920CAMERAIN_HPP__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "mediain.hpp"

#ifdef __linux__
#include "../C920VideoCap/C920Camera.h"
#endif

// Code specific for C920 camera. We have lots of
// extra controls avaiable for this, so use it if
// possible
class C920CameraIn : public MediaIn
{
   public:
      C920CameraIn(int _stream = -1, bool gui = false);
	  ~C920CameraIn() {}
      bool update();
      bool getFrame(cv::Mat &frame, cv::Mat &depth, bool pause = false);

      int width(void) const;
      int height(void) const;

	  int frameNumber(void) const;

#ifdef __linux__
	  CameraParams getCameraParams(bool left) const;
#endif

   private:
#ifdef __linux__
      bool initCamera(bool gui);

	  // Mark these as friends so they can access private class data
	  friend void brightnessCallback(int value, void *data);
	  friend void contrastCallback(int value, void *data);
	  friend void saturationCallback(int value, void *data);
	  friend void sharpnessCallback(int value, void *data);
	  friend void gainCallback(int value, void *data);
	  friend void autoExposureCallback(int value, void *data);
	  friend void backlightCompensationCallback(int value, void *data);
	  friend void whiteBalanceTemperatureCallback(int value, void *data);
	  friend void focusCallback(int value, void *data);

      v4l2::C920Camera  camera_;
	  cv::Mat           localFrame_;
      int               brightness_;
      int               contrast_;
      int               saturation_;
      int               sharpness_;
      int               gain_;
      int               focus_;
      int               autoExposure_;
      int               backlightCompensation_;
      int               whiteBalanceTemperature_;
	  int               frameNumber_;
	  int               lockedFrameNumber_;
	  v4l2::CaptureSize captureSize_;
#endif
};
#endif
