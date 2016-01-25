#ifndef INC__GROUNDTRUTH__HPP__
#define INC__GROUNDTRUTH__HPP__

#include <map>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

// Simple class to store ground truth data. A ground truth entry is just a 
// known location of the object we're detecting in a video. Here, they're 
// stored as a video name, frame number and location rectangle.
class GroundTruth
{
	public :
		GroundTruth(const std::string &truthFile, const std::string &videoFile);
		std::vector<cv::Rect> get(unsigned int frame) const;
		std::vector<unsigned int> getFrameList(void) const;

	private :
		std::map< unsigned int, std::vector<cv::Rect> > map_;
};

#endif
