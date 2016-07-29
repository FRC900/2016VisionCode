#pragma once

#include <opencv2/opencv.hpp>

cv::Rect ResizeRect(const cv::Rect& rect, const cv::Size& size);
cv::Rect AdjustRect(const cv::Rect& rect, const double ratio);
bool RescaleRect(const cv::Rect& inRect, cv::Rect& outRect, const cv::Size& imageSize, const double scaleUp);

void rotateImageAndMask(const cv::Mat &srcImg, const cv::Mat &srcMask,
						const cv::Scalar &bgColor, const cv::Point3f &maxAngle,
						cv::RNG &rng, cv::Mat &outImg, cv::Mat &outMask);
