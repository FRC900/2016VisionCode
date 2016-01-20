#include <stdint.h>
#include "fast.hpp"

namespace fovis
{

	void FAST(uint8_t* image, uint8_t* mask, int width, int height, int row_stride,
		std::vector<KeyPoint>* keypoints, int threshold, bool nonmax_suppression)
	{
	   
	   std::vector<cv::KeyPoint> cvKeypoints;
	   cv::Mat frameCPU(height,width,CV_8UC1,image);
	   cv::gpu::GpuMat frameGPU(height,width,CV_8UC1); //initialize mats

	   frameGPU.upload(frameCPU); //copy frame from cpu to gpu

	   cv::gpu::GpuMat maskCV(height,width,CV_8UC1,mask);

	   cv::gpu::FAST_GPU FASTObject(threshold,nonmax_suppression); //run the detection
	   FASTObject(frameGPU,maskCV,cvKeypoints);

	   keypoints->clear();
	   for(uint i = 0; i < cvKeypoints.size(); i++) { //the pointers here are so ugly please
		keypoints->push_back(fovis::KeyPoint()); //don't read these lines
		(*keypoints)[i].u = cvKeypoints[i].pt.x; //just follow the comments
		(*keypoints)[i].v = cvKeypoints[i].pt.y; //because if you read them all the 
		(*keypoints)[i].score = cvKeypoints[i].response; //way you can skip reading this section of code
		}
	}

}
