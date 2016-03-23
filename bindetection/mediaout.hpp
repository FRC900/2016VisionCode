#pragma once

#include <boost/thread.hpp>
#include <opencv2/core/core.hpp>

// Base class for output.  Derived classes are for writing 
// AVI videos, zms (video + depth), plus whatever else we
// imagine in the future.
class MediaOut
{
   public:
		MediaOut(int frameSkip, int framesPerFile);
		virtual ~MediaOut();
		bool saveFrame(const cv::Mat &frame, const cv::Mat &depth);

   protected:
		// The base class calls these dervied classes to do the 
		// heavy lifting.  They have to be implemented in the 
		// base class as well, but hopefully those are never
		// called 
		virtual bool openNext(void);
		virtual bool write(const cv::Mat &frame, const cv::Mat &depth);

		// Skip output frames if requested.  Skip is how many to 
		// skip before writing next output frame, FrameCounter is how
		// many total frames seen.
		// Counter is used to split the output into multiple shorter
		// outputs
		int frameSkip_;
		int frameCounter_;
		int fileCounter_;
		int framesPerFile_;

   private: 
		void writeThread(void);
		cv::Mat frame_;
		cv::Mat depth_;
		boost::mutex matLock_;
		boost::mutex fileLock_;
		boost::condition_variable frameCond;
		bool frameReady_;
		boost::thread thread_;
};
