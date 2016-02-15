#ifndef INC_DETECT_HPP__
#define INC_DETECT_HPP__

#include "CaffeBatchPrediction.hpp"

// Turn Window from a typedef into a class :
//   Private members are the rect, index from Window plus maybe a score?
//   Constructor takes Rect, size_t index
//   Maybe another constructor with Rect, size_t scaleIndex, float score
//   Need calls for 
//      - get score
//      - get rect
//      - rescaling rect by a fixed value - multiply or divide the x,y,
//        width, height by a passed-in constant
//      - get scaled rect - given a scaledImages array, return the rect
//        scaled back to fit correctly on the original image. Should be
//        something like double scale = scaledImages[index], 
//        return rect(scaled down by scale.  See the first for loop in runNMS
//        for an example of this
//      - get an image for the window. Pass in scaledImages and a Mat. 
//        Fill in the Mat with the image data pulled from the correct scaled
//        image (the one from the entry <index> in the scaledImage array).
//        See the top of the loop in runDetection for an example
//

template <class MatT>
class NNDetect
{
	public:
		NNDetect(const std::vector<std::string> &d12Info,
				 const std::vector<std::string> &d24Info) :
			d12_(CaffeClassifier<MatT>(d12Info[0], d12Info[1], d12Info[2], d12Info[3], 64)),
			d24_(CaffeClassifier<MatT>(d24Info[0], d24Info[1], d24Info[2], d24Info[3], 64))
		{
		}
		void detectMultiscale(const cv::Mat &inputImg,
				cv::Mat &depthIn,
				const cv::Size &minSize,
				const cv::Size &maxSize,
				double scaleFactor,
				double nmsThreshold,
				const std::vector<double> &detectThreshold,
				std::vector<cv::Rect> &rectsOut);

	private:
		typedef std::pair<cv::Rect, size_t> Window;
		CaffeClassifier <MatT> d12_;
		CaffeClassifier <MatT> d24_;
		void doBatchPrediction(CaffeClassifier<MatT> &classifier,
				const std::vector<MatT> &imgs,
				float threshold,
				const std::string &label,
				std::vector<size_t> &detected,
				std::vector<float>  &scores);

		void generateInitialWindows(
				const MatT &input,
				cv::Mat &depthIn,
				const cv::Size &minSize,
				const cv::Size &maxSize,
				int wsize,
				double scaleFactor,
				std::vector<std::pair<MatT, double> > &scaledimages,
				std::vector<Window> &windows);

		void runDetection(CaffeClassifier<MatT> &classifier,
				const std::vector<std::pair<MatT, double> > &scaledimages,
				const std::vector<Window> &windows,
				float threshold,
				std::string label,
				std::vector<Window> &windowsOut,
				std::vector<float> &scores);

		void runNMS(const std::vector<Window> &windows, 
				const std::vector<float> &scores,  
				const std::vector<std::pair<MatT, double> > &scaledImages,
				double nmsThreshold,
				std::vector<Window> &windowsOut);

		bool depthInRange(float depth_min, float depth_max, cv::Mat &detectCheck);
};

#endif
