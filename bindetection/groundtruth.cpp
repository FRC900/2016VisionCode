#include <fstream>
#include <iostream>

#include "groundtruth.hpp"

using namespace std;
using namespace cv;

GroundTruth::GroundTruth(const string &truthFile, const string &videoFile)
{
	ifstream ifs(truthFile, ifstream::in);
	Rect rect;
	int frame;
	string videoName;

	while (ifs >> videoName >> frame >> rect.x >> rect.y >> rect.width >> rect.height)
	{
		if (videoName == videoFile.substr(videoFile.find_last_of("\\/") + 1))
		{
			map_[frame].push_back(rect);
		}
	}
}


std::vector<cv::Rect> GroundTruth::get(unsigned int frame) const
{
	map<unsigned int, vector<Rect> >::const_iterator it = map_.find(frame);
	if (it == map_.end())
		return vector<Rect>();

	return it->second;
}
