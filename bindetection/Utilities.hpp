//standard include
#include <math.h>
#include <iostream>
#include <fstream>

//#include <Eigen/Geometry>
#include <cmath>

//opencv include
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

namespace utils {

cv::Point3f screenToWorldCoords(const cv::Point &screen_position, double avg_depth, const cv::Point2f &fov_size, const cv::Size &frame_size, float cameraElevation);

std::pair<float, float> minOfDepthMat(const cv::Mat& img, const cv::Mat& mask, const cv::Rect& bound_rect, int range);
void shrinkRect(cv::Rect &rect_in, float shrink_factor);

std::pair<double,double> slopeOfMasked(const cv::Mat &depth, const cv::Mat &mask, cv::Point2f fov);
//void printIsometry(const Eigen::Transform<double, 3, Eigen::Isometry> m);

double normalCFD(const std::pair<double, double> &meanAndStdev, double value);

}
