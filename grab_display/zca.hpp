#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class ZCA
{
	public:
		ZCA(const std::vector<cv::Mat> &images, const cv::Size &size, float epsilon = 0.00001);
		ZCA(const char *xmlFilename);

		cv::Mat Transform8UC3(const cv::Mat &input) const;
		cv::Mat Transform32FC3(const cv::Mat &input) const;
		void Write(const char *xmlFilename) const;

	private:
		cv::Size size_;
		cv::Mat  weights_;
		float    epsilon_;
};
