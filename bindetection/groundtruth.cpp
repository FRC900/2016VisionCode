#include <fstream>
#include <algorithm>

#include "groundtruth.hpp"

using namespace std;
using namespace cv;

// Constructor - read from the file on disk truthFile, grabbing only
// ground truth information for the input video videoFile.
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
			// Creates an entry if not there.  If one is already
			// there, just reuse the vector as is - this allows
			// for the possibility of mulitple rects per frame
			// if multiple copies of the object are present
			map_[frame].push_back(rect);
		}
	}
}


// Grab the list of ground truths for a given frame
std::vector<cv::Rect> GroundTruth::get(unsigned int frame) const
{
	map<unsigned int, vector<Rect> >::const_iterator it = map_.find(frame);
	if (it == map_.end())
		return vector<Rect>();

	return it->second;
}

// Get a sorted list of frames with valid ground truth data
std::vector<unsigned int> GroundTruth::getFrameList(void) const
{
	vector<unsigned int> retVec;
	for (map<unsigned int, vector<Rect> >::const_iterator it = map_.begin(); it != map_.end(); ++it)
		retVec.push_back(it->first);

	sort(retVec.begin(), retVec.end());
	return retVec;
}
