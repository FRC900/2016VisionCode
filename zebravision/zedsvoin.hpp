#pragma once
// Use ZED SVO input files. Similar to other video in classes.
// Runs an update thread in the background which tries to
// decode one frame ahead of the frame used in getFrame().

#include <boost/thread.hpp>
#include <opencv2/core/core.hpp>
#include <zed/Camera.hpp>

#include "syncin.hpp"
#include "zedparams.hpp"

class ZedSVOIn : public SyncIn
{
	public:
		ZedSVOIn(const char *inFileName = NULL, ZvSettings *settings = NULL);
		~ZedSVOIn();

#ifdef ZED_SUPPORT
		bool isOpened(void) const;

		// How many frames?
		int    frameCount(void) const;

	protected:
		// Defined in derived classes to handle the nuts
		// and bolts of grabbing a frame from a given
		// source.  preLock happens before the mutex
		// while postLock happens inside it
		bool postLockUpdate(cv::Mat &frame, cv::Mat &depth);
		bool postLockFrameNumber(int framenumber);

	private:
		sl::zed::Camera *zed_;
		ZedParams        params_;
#endif
};
