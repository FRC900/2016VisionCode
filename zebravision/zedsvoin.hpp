#pragma once

//opencv include
#include <opencv2/core/core.hpp>
#include "zedin.hpp"

class ZedSVOIn : public ZedIn
{
	public:
		ZedSVOIn(const char *inFileName = NULL, ZvSettings *settings = NULL);
		~ZedSVOIn();

#ifdef ZED_SUPPORT
		bool isOpened(void) const;
		bool update(void);
		bool getFrame(cv::Mat &frame, cv::Mat &depth, bool pause = false);

		// How many frames?
		int    frameCount(void) const;

		// Get and set current frame number
		int    frameNumber(void) const;
		void   frameNumber(int frameNumber);

	private:
		bool update(bool left);
#endif
};
