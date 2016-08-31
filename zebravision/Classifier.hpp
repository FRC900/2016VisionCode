#pragma once
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include <opencv2/core/core.hpp>

#include "zca.hpp"

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<std::string, float> Prediction;

class Classifier 
{
	public:
		Classifier(const std::string& modelFile,
					const std::string& trainedFile,
					const std::string& zcaWeightFile,
					const std::string& labelFile,
					const size_t batchSize);

		// Given X input images, return X vectors of predictions.
		// Each prediction is a label, value pair, where the value is
		// the confidence level for each given label.
		// Each of the X vectors are themselves a vector which will have the 
		// N predictions with the highest confidences for the corresponding
		// input image
		std::vector< std::vector<Prediction> > ClassifyBatch(const std::vector< cv::Mat > &imgs, size_t numClasses);

		// Get the width and height of an input image to the net
		cv::Size getInputGeometry(void) const;

		// Get the batch size of the model
		size_t BatchSize(void) const;

	protected:
		bool fileExists(const std::string &fileName) const;
		cv::Size inputGeometry_;          // size of one input image
		size_t batchSize_;                // number of images to process in one go
		ZCA  zca_;                        // weights used to normalize input data
		std::vector<std::string> labels_; // labels for each output index

	private:
		// Get the output values for a set of images
		// These values will be in the same order as the labels for each
		// image, and each set of labels for an image next adjacent to the
		// one for the next image.
		// That is, [0] = value for label 0 for the first image up to 
		// [n] = value for label n for the first image. It then starts again
		// for the next image - [n+1] = label 0 for image #2.
		virtual std::vector<float> PredictBatch(const std::vector<cv::Mat> &imgs);

};
