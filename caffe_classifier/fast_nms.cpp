/**
 * Fast non-maximum suppression in C, port from  
 * http://quantombone.blogspot.com/2011/08/blazing-fast-nmsm-from-exemplar-svm.html
 *
 * @blackball (bugway@gmail.com)
 */

#include <algorithm>
#include <iostream>

#include "fast_nms.hpp"

// TODO : see if these hacks really are faster or not
#define fast_max(x,y) (x - ((x - y) & ((x - y) >> (sizeof(int) * CHAR_BIT - 1))))
#define fast_min(x,y) (y + ((x - y) & ((x - y) >> (sizeof(int) * CHAR_BIT - 1))))

static bool detectedCompareGreater(const Detected &rh, const Detected &lh)
{
   return rh.second > lh.second;
}

void 
fastNMS(std::vector<Detected> detected, float overlap_th, std::vector<cv::Rect> &filteredList) 
{
   filteredList.clear(); // Clear out return array

   // Sort input rects by decreasing score - i.e. look at best
   // values first
   std::sort(detected.begin(), detected.end(), detectedCompareGreater);

   std::vector<bool> validList(detected.size(), true);

   // Loop while there's anything valid left in rects array
   bool anyValid = true;
   do
   {
      anyValid = false; // assume there's nothing valid, adjust later if needed

      // Look for first valid entry in rects
      size_t i;
      for (i = 0; i < validList.size(); ++i)
	 if (validList[i])
	    break;

      // Exit if none are found
      if (i == validList.size())
	 break;

      // Save the highest ranked remaining DRect
      // and invalidate it - this means we've already
      // processed it
      filteredList.push_back(detected[i].first);
      validList[i] = false;

      // Save coords of this DRect so we can
      // filter out nearby DRects which have a lower
      // ranking
      int x0 = detected[i].first.tl().x;
      int y0 = detected[i].first.tl().y;
      int x1 = detected[i].first.br().x;
      int y1 = detected[i].first.br().y;

      // Loop through the rest of the array, looking
      // for entries which overlap with the current "good"
      // one being processed
      for (++i; i < detected.size() ; ++i) 
      {
	 if (validList[i])
	 {
	    int tx0 = fast_max(x0, detected[i].first.tl().x);
	    int ty0 = fast_max(y0, detected[i].first.tl().y);
	    int tx1 = fast_min(x1, detected[i].first.br().x);
	    int ty1 = fast_min(y1, detected[i].first.br().y);

	    tx0 = tx1 - tx0 + 1;
	    ty0 = ty1 - ty0 + 1;
	    if ((tx0 > 0) && (ty0 > 0) && 
		((tx0 * ty0 / (float)detected[i].first.area()) > overlap_th)) 
	       validList[i] = false; // invalidate DRects which overlap
	    else
	       anyValid = true;  // otherwise indicate that there's stuff left to do next time
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

