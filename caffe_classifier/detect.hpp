#ifndef INC_DETECT_HPP__
#define INC_DETECT_HPP__

#include "CaffeBatchPrediction.hpp"

template <class MatT>
class NNDetect
{
	private:
	CaffeClassifier <MatT> classifier;
	void doBatchPrediction(CaffeClassifier<MatT> &classifier,
			const std::vector<MatT> &imgs,
			const float threshold,
			const std::string &label,
			std::vector<size_t> &detected);
	void generateInitialWindows(
			const cv::Mat  &input,
			const cv::Size &minSize,
			const cv::Size &maxSize,
			const int wsize,
			double scaleFactor,
			std::vector<std::pair<MatT, float> > &scaledimages,
			std::vector<cv::Rect> &rects,

std::vector<int> &scales);
	void runDetection(CaffeClassifier<MatT> &classifier,
			const std::vector<std::pair<MatT, float> > &scaledimages,
			const std::vector<cv::Rect> &rects,
			const std::vector<int> &scales,
			float threshold,
			std::string label,
			std::vector<cv::Rect> &rectsOut,
			std::vector<int> &scalesOut);

	public:
	NNDetect(const std::string &model_file,
		const std::string &trained_file,
		const std::string &mean_file,
		const std::string &label_file):
	classifier(CaffeClassifier<MatT>(model_file, trained_file, mean_file, label_file, 64 ))
	{
	}
	void detectMultiscale(const cv::Mat &inputImg,
		const cv::Size &minSize,
		const cv::Size &maxSize,
		double          scaleFactor,
		std::vector<cv::Rect> &rectsOut);
};

#endif
