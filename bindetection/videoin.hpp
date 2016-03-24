#ifndef VIDEOIN_HPP__
#define VIDEOIN_HPP__

#include <opencv2/highgui/highgui.hpp>
#include "mediain.hpp"

class VideoIn : public MediaIn
{
	public:
		VideoIn(const char *inpath);
		~VideoIn() {}
		bool update(void);
		bool getFrame(cv::Mat &frame, cv::Mat &depth, bool pause = false);
		int width(void) const;
		int height(void) const;
		int frameCount(void) const;
		int frameNumber(void) const;
		void frameNumber(int frameNumber);

	private:
		cv::VideoCapture cap_;
		int              width_;
		int              height_;
		int              frames_;
		int              frameNumber_;
};
#endif
