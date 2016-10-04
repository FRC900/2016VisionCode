#ifndef CAMERAIN_HPP__
#define CAMERAIN_HPP__

#include <opencv2/highgui/highgui.hpp>
#include "mediain.hpp"

class ZvSettings;

class CameraIn : public MediaIn
{
	public:
		CameraIn(int stream = -1, ZvSettings *settings = NULL);
		~CameraIn();
		bool isOpened(void) const;
		bool update(void);
		bool getFrame(cv::Mat &frame, cv::Mat &depth, bool pause = false);

	private:
		int              saveWidth_;
		int              saveHeight_;
		double           fps_;
		cv::Mat          localFrame_;
		cv::Mat          pausedFrame_;
		cv::VideoCapture cap_;

		std::string getClassName() const { return "CameraIn"; } 
		bool loadSettings(void);
		bool saveSettings(void) const;
};
#endif
