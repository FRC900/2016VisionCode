#ifndef INC_SCALEFACTOR_HPP__
#define INC_SCALEFACTOR_HPP__

#include "opencv2_3_shim.hpp"
#include <vector>

void scalefactor(const cv::Mat &inputimage, const cv::Size &objectsize, 
      const cv::Size &minsize, const cv::Size &maxsize, double scaleFactor, 
      std::vector<std::pair<cv::Mat, double> > &ScaleInfo);

void scalefactor(const cv::Mat &inputimage, 
		const std::vector<std::pair<cv::Mat, double> > &scaleInfoIn, 
		int rescaleFactor, 
		std::vector<std::pair<cv::Mat, double> > &scaleInfoOut);

void scalefactor(const GpuMat &inputimage, const cv::Size &objectsize, 
      const cv::Size &minsize, const cv::Size &maxsize, double scaleFactor, 
      std::vector<std::pair<GpuMat, double> > &ScaleInfo);

void scalefactor(const GpuMat &inputimage, 
		const std::vector<std::pair<GpuMat, double> > &scaleInfoIn, 
		int rescaleFactor, 
		std::vector<std::pair<GpuMat, double> > &scaleInfoOut);

#endif
