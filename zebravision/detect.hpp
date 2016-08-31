#ifndef INC_DETECT_HPP__
#define INC_DETECT_HPP__

#include "Classifier.hpp"

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
		NNDetect(Classifier<MatT> *d12,
			     Classifier<MatT> *d24, 
	   		     Classifier<MatT> *c12,
			     Classifier<MatT> *c24, 
			     float hfov)  :
			d12_(d12),
			d24_(d24),
			c12_(c12),
			c24_(c24),
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
				std::vector<cv::Rect> &rectsOut,
				std::vector<cv::Rect> &uncalibRectsOut);

	private:
		typedef std::pair<cv::Rect, size_t> Window;
		Classifier<MatT> *d12_;
		Classifier<MatT> *d24_;
		Classifier<MatT> *c12_;
		Classifier<MatT> *c24_;
		float hfov_;
		void doBatchPrediction(Classifier<MatT> *&classifier,
				const std::vector<MatT> &imgs,
				float threshold,
				const std::string &label,
				std::vector<size_t> &detected,
				std::vector<float>  &scores);

		void generateInitialWindows(
				const MatT &input,
				const MatT &depthIn,
				const cv::Size &minSize,
				const cv::Size &maxSize,
				int wsize,
				double scaleFactor,
				std::vector<std::pair<MatT, double> > &scaledimages,
				std::vector<Window> &windows);

		void runDetection(Classifier<MatT> *&classifier,
				const std::vector<std::pair<MatT, double> > &scaledimages,
				const std::vector<Window> &windows,
				float threshold,
				std::string label,
				std::vector<Window> &windowsOut,
				std::vector<float> &scores);

		void runGlobalNMS(const std::vector<Window> &windows, 
				const std::vector<float> &scores,  
				const std::vector<std::pair<MatT, double> > &scaledImages,
				double nmsThreshold,
				std::vector<Window> &windowsOut);
		void runLocalNMS(const std::vector<Window> &windows, 
				const std::vector<float> &scores,  
				double nmsThreshold,
				std::vector<Window> &windowsOut);
		void runCalibration(const std::vector<Window>& windowsIn,
				    const std::vector<std::pair<MatT, double> > &scaledImages,
				    Classifier<MatT> *&classifier,
				    float threshold,
				    std::vector<Window>& windowsOut);
		void doBatchCalibration(Classifier<MatT> *&classifier,
					const std::vector<MatT>& imags,
					float threshold,
					std::vector<std::vector<float> >& shift);
		bool depthInRange(float depth_min, float depth_max, const MatT &detectCheck);
};

#endif
