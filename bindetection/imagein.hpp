#ifndef IMAGEIN_HPP__
#define IMAGEIN_HPP__

#include <opencv2/core/core.hpp>

#include "mediain.hpp"

// Still image (png, jpg) processing
class ImageIn : public MediaIn
{
   public:
      ImageIn(const char *outpath);
	  ~ImageIn() {}
      bool update();
      bool getFrame(cv::Mat &frame, cv::Mat &depth);
	  int semValue(void) { return 1; }

	  int frameCount(void) const;
	  int frameNumber(void) const;

      int width() const;
      int height() const;
  private:
    std::string outpath_;
};
#endif
