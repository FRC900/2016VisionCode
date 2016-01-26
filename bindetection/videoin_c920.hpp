#ifndef VIDEOIN_C920_HPP__
#define VIDEOIN_C920_HPP__

// video 4 linux code doesn't work on cygwin,
// so fall back to normal OpenCV videocapture code
#ifndef __linux
#include "videoin.hpp"
#else

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "../C920VideoCap/C920Camera.h"

class VideoIn
{
   public:
      VideoIn(const char *path);
      VideoIn(int _stream = -1, bool gui = false);
	  bool initCamera(int _stream, bool gui);

      cv::VideoCapture *VideoCap(void);
      bool getNextFrame(bool pause, cv::Mat &frame);
      int frameCounter(void);
      void frameCounter(int frameCount);

	  // Mark these as friends so they can access private class data
	  friend void brightnessCallback(int value, void *data);
	  friend void contrastCallback(int value, void *data);
	  friend void saturationCallback(int value, void *data);
	  friend void sharpnessCallback(int value, void *data);
	  friend void gainCallback(int value, void *data);
	  friend void backlightCompensationCallback(int value, void *data);
	  friend void whiteBalanceTemperatureCallback(int value, void *data);
	  friend void focusCallback(int value, void *data);
   private:
      v4l2::C920Camera _camera;
      cv::VideoCapture _cap;
	  v4l2::CaptureSize _captureSize;
      cv::Mat          _frame;
      int              _frameCounter;
      bool             _c920;
      bool             _video;
      int              _brightness;
      int              _contrast;
      int              _saturation;
      int              _sharpness;
      int              _gain;
      int              _focus;
      int              _backlightCompensation;
      int              _whiteBalanceTemperature;
};
#endif
#endif

