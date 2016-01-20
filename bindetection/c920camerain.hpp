#ifndef C920CAMERAIN_HPP__
#define C920CAMERAIN_HPP__

#include "opencv2/highgui/highgui.hpp"
#include "mediain.hpp"

#ifdef __linux__
#include "../C920VideoCap/C920Camera.h"
#endif

class C920CameraIn : public MediaIn
{
   public:
      C920CameraIn(int _stream = -1, bool gui = false);
      bool getNextFrame(cv::Mat &frame, bool pause = false);

      int width(void);
      int height(void);

   private:
#ifdef __linux__
      bool initCamera(int _stream, bool gui);

	  friend void brightnessCallback(int value, void *data);

      v4l2::C920Camera  _camera;
      cv::Mat           _frame;
      int               _brightness;
      int               _contrast;
      int               _saturation;
      int               _sharpness;
      int               _gain;
      int               _focus;
      int               _backlightCompensation;
      int               _whiteBalanceTemperature;
	  int               _frameCounter;
	  v4l2::CaptureSize _captureSize;
#endif
};
#endif

