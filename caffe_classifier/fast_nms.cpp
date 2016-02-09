/**
 * Fast non-maximum suppression in C, port from  
 * http://quantombone.blogspot.com/2011/08/blazing-fast-nmsm-from-exemplar-svm.html
 *
 * @blackball (bugway@gmail.com)
 */

#include <algorithm>
#include <functional>
#include <iostream>

#include "fast_nms.hpp"

class DetectedPlusIndex
{
	public :
		DetectedPlusIndex(const cv::Rect &rect, double score, size_t index) :
			rect_(rect),
			score_(score),
			index_(index)
	{}
		cv::Rect rect_;
		float    score_;
		size_t   index_;

		bool operator> (const DetectedPlusIndex &other) const
		{
			return score_ > other.score_;
		}
};

void fastNMS(const std::vector<Detected> &detected, double overlap_th, std::vector<size_t> &filteredList) 
{
	filteredList.clear(); // Clear out return array
	std::vector <DetectedPlusIndex> dpi;

	// Create a list that includes the detected input plus the
	// index into the list as it was passed in.  Keep the index
	// so we can pass the original unsorted index back to the caller.
	size_t idx = 0;
	for (auto it = detected.cbegin(); it != detected.cend(); ++it)
		dpi.push_back(DetectedPlusIndex(it->first, it->second, idx++));

	// Sort input rects by decreasing score - i.e. look at best
	// values first
	std::sort(dpi.begin(), dpi.end(), std::greater<DetectedPlusIndex>());
	for (auto it = dpi.cbegin(); it != dpi.cend(); ++it)
		std::cerr << it->rect_<< " " << it->score_ << " " << it->index_ << std::endl;

	// Start by assuming all of the input rects are non-overlapping
	std::vector<bool> validList(dpi.size(), true);

	// Loop while there's anything valid left in rects array
	bool anyValid;
	do
	{
		anyValid = false; // assume there's nothing valid, adjust later if needed

		// Look for the highest scoring unprocessed 
		// rectangle left in the list
		size_t i;
		for (i = 0; i < validList.size(); ++i)
			if (validList[i])
				break;

		// Exit if everything has been processed
		if (i == validList.size())
			break;

		// Save the index of the highest ranked remaining Rect
		// and invalidate it - this means we've already
		// processed it
		filteredList.push_back(dpi[i].index_);
		validList[i] = false;

		// Save this rect to compare against the
		// remaining lower-scoring ones
		cv::Rect topRect = dpi[i].rect_;

		// Loop through the rest of the array, looking
		// for entries which overlap with the current "good"
		// one being processed
		for (++i; i < dpi.size(); ++i) 
		{
			// Only check entries which haven't 
			// been removed already
			if (validList[i])
			{
				cv::Rect thisRect = dpi[i].rect_;

				// Look at the Intersection over Union ratio.
				// The higher this is, the closer the two rects are
				// to overlapping
				double intersectArea = (topRect & thisRect).area();
				double unionArea     = topRect.area() + thisRect.area() - intersectArea;

				if ((intersectArea / unionArea) <= overlap_th)
					validList[i] = false; // invalidate Rects which overlap
				else
					anyValid = true;      // otherwise indicate that there's stuff left to do next time
			}
		}
	}
	while (anyValid);
}

#if 0
	static void 
test_nn() 
{
	std::vector<Detected> rects;
	std::vector<cv::Rect> keep;

	rects.push_back(Detected(cv::Rect(cv::Point(0,  0),  cv::Point(10+1, 10+1)), 0.5f));
	rects.push_back(Detected(cv::Rect(cv::Point(1,  1),  cv::Point(10+1, 10+1)), 0.4f));
	rects.push_back(Detected(cv::Rect(cv::Point(20, 20), cv::Point(40+1, 40+1)), 0.3f));
	rects.push_back(Detected(cv::Rect(cv::Point(20, 20), cv::Point(40+1, 30+1)), 0.4f));
	rects.push_back(Detected(cv::Rect(cv::Point(15, 20), cv::Point(40+1, 40+1)), 0.1f));

	fastNMS(rects, 0.4f, keep);

	for (size_t i = 0; i < keep.size(); i++)
		std::cout << keep[i] << std::endl;
}

	int 
main(int argc, char *argv[]) 
{
	test_nn();
	return 0;
}
#endif

