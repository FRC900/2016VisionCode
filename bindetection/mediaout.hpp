#pragma once

#include <opencv2/core/core.hpp>

// Base class for output.  Derived classes are for writing 
// AVI videos, zms (video + depth), plus whatever else we
// imagine in the future.
class MediaOut
{
   public:
		MediaOut();
		virtual ~MediaOut();
		virtual bool saveFrame(const cv::Mat &frame, const cv::Mat &depth) = 0;

   private:
};
