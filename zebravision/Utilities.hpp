#pragma once
//standard include
#include <fstream>
#include <iostream>
#include <string>

//#include <Eigen/Geometry>

//opencv include
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

namespace utils {

std::pair<float, float> minOfDepthMat(const cv::Mat& img, const cv::Mat& mask, const cv::Rect& bound_rect, int range);
void shrinkRect(cv::Rect &rect_in, float shrink_factor);

//void printIsometry(const Eigen::Transform<double, 3, Eigen::Isometry> m);

double normalCFD(const std::pair<double, double> &meanAndStdev, double value);

bool fileExists(const std::string &fileName);

}
