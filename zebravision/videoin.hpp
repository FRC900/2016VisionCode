#ifndef VIDEOIN_HPP__
#define VIDEOIN_HPP__

#include <opencv2/highgui/highgui.hpp>
#include "mediain.hpp"

class ZvSettings;

class VideoIn : public MediaIn
{
	public:
		VideoIn(const char *inpath, ZvSettings *settings = NULL);
		~VideoIn() {}
		bool isOpened(void) const;
		bool update(void);
		bool getFrame(cv::Mat &frame, cv::Mat &depth, bool pause = false);
		int  width(void) const;
		int  height(void) const;
		int  frameCount(void) const;
		void frameNumber(int frameNumber);

	private:
		cv::VideoCapture cap_;
		int              frames_;
		bool             frameReady_;
		boost::condition_variable condVar_;

		bool loadSettings(void) { return true; }
		bool saveSettings(void) const { return true; }
		std::string getClassName() const { return "VideoIn"; }
};
#endif
