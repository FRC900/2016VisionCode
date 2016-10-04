#ifndef IMAGEIN_HPP__
#define IMAGEIN_HPP__

#include <opencv2/core/core.hpp>

#include "mediain.hpp"

class ZvSettings;

// Still image (png, jpg) processing
class ImageIn : public MediaIn
{
	public:
		ImageIn(const char *outpath, ZvSettings *settings = NULL);
		~ImageIn() {}
		bool isOpened(void) const;
		bool update(void);
		bool getFrame(cv::Mat &frame, cv::Mat &depth, bool pause = false);

		int frameCount(void) const;

	private:
		std::string outpath_;
		std::string getClassName() const { return "ImageIn"; }
		bool loadSettings(void) { return true; }
		bool saveSettings(void) const { return true; }
};
#endif
