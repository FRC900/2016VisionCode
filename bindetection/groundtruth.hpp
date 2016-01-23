#ifndef INC__GROUNDTRUTH__HPP__
#define INC__GROUNDTRUTH__HPP__

#include <map>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

class GroundTruth
{
	public :
		GroundTruth(const std::string &truthFile, const std::string &videoFile);
		std::vector<cv::Rect> get(unsigned int frame) const;

	private :
		std::map< unsigned int, std::vector<cv::Rect> > map_;
};

#endif
