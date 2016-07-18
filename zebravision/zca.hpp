#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

class ZCA
{
	public:
		ZCA(const std::vector<cv::Mat> &images, const cv::Size &size, float epsilon = 0.00001);
		ZCA(const char *xmlFilename);

		// Apply ZCA transofrm to a single image in
		// 8UC3 format (normal imread) and 32FC3
		// format (3 channels of float input, values 
		// scaled between 0 and 1)
		cv::Mat Transform8UC3 (const cv::Mat &input);
		cv::Mat Transform32FC3(const cv::Mat &input);

		// Batch versions of above
		std::vector<cv::Mat>Transform8UC3 (const std::vector<cv::Mat> &input);
		std::vector<cv::Mat>Transform32FC3(const std::vector<cv::Mat> &input);
		void Write(const char *xmlFilename) const;

	private:
		cv::Size size_;
		cv::Mat          weights_;
		cv::gpu::GpuMat  weightsGPU_;
		cv::gpu::GpuMat  gm_;
		cv::gpu::GpuMat  gmOut_;

		cv::gpu::GpuMat  buf_;
		float    epsilon_;
};
