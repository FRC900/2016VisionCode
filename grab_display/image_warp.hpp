#pragma once

#include <opencv2/opencv.hpp>

void rotateImageAndMask(const cv::Mat &srcImg, const cv::Mat &srcMask,
						const cv::Scalar &bgColor, const cv::Point3f &maxAngle,
						cv::RNG &rng, cv::Mat &outImg, cv::Mat &outMask);
