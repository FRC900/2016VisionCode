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
			 const std::vector<std::string> &d24Info, 
			const std::vector<std::string> &c12Info,
			const std::vector<std::string> &c24Info, 
			float hfov)  :
			d12_(CaffeClassifier<MatT>(d12Info[0], d12Info[1], d12Info[2], d12Info[3], 64)),
			d24_(CaffeClassifier<MatT>(d24Info[0], d24Info[1], d24Info[2], d24Info[3], 64)),
			c12_(CaffeClassifier<MatT>(c12Info[0], c12Info[1], c12Info[2], c12Info[3], 64)),
			c24_(CaffeClassifier<MatT>(c24Info[0], c24Info[1], c24Info[2], c24Info[3], 64)),
			hfov_(hfov)
		{
		}
		void detectMultiscale(const cv::Mat &inputImg,
				const cv::Mat &depthIn,
				const cv::Size &minSize,
				const cv::Size &maxSize,
				double scaleFactor,
				const std::vector<double> &nmsThreshold,
				const std::vector<double> &detectThreshold,
				const std::vector<double> &calThreshold,
				std::vector<cv::Rect> &rectsOut);

	private:
		typedef std::pair<cv::Rect, size_t> Window;
		CaffeClassifier <MatT> d12_;
		CaffeClassifier <MatT> d24_;
		CaffeClassifier <MatT> c12_;
		CaffeClassifier <MatT> c24_;
		float hfov_;
		void doBatchPrediction(CaffeClassifier<MatT> &classifier,
				const std::vector<MatT> &imgs,
				float threshold,
				const std::string &label,
				std::vector<size_t> &detected,
				std::vector<float>  &scores);

		void generateInitialWindows(
				const MatT &input,
				const cv::Mat &depthIn,
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
		
		void runCalibration(const std::vector<Window>& windowsIn,
				    const std::vector<std::pair<MatT, double> > &scaledImages,
				    CaffeClassifier<MatT>& classifier,
				    float threshold,
				    std::vector<Window>& windowsOut);

		void doBatchCalibration(CaffeClassifier<MatT>& classifier,
					const std::vector<MatT>& imags,
					float threshold,
					std::vector<std::vector<float> >& shift);
		bool depthInRange(float depth_min, float depth_max, const cv::Mat &detectCheck);
};

#endif
