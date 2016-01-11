#ifndef INC_FAST_NMS_H__
#define INC_FAST_NMS_H__

#include <opencv2/core/core.hpp>
#include <utility>
#include <vector>

typedef std::pair<cv::Rect, float> Detected;
void fastNMS(std::vector<Detected> detects, float overlap_th, std::vector<cv::Rect> &filteredList);
#endif
