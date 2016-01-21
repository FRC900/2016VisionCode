#ifndef __fovis_FAST_hpp__
#define __fovis_FAST_hpp__

#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "keypoint.hpp"

namespace fovis
{

void FAST(uint8_t* img, int width, int height, int row_stride,
    std::vector<KeyPoint>* keypoints, 
    int threshold, 
    bool nonmax_suppression);

}

#endif
