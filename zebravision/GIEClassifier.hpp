#pragma once

#include "Infer.h"
#include "Classifier.hpp"

class GIEClassifier : public Classifier
{
	public:
		GIEClassifier(const std::string& modelFile,
					const std::string& trainedFile,
					const std::string& zcaWeightFile,
					const std::string& labelFile,
					const size_t batchSize);
		~GIEClassifier();

	private:
		// Wrap input layer of the net into separate Mat objects
		// This sets them up to be written with actual data
		// in PreprocessBatch()
		void WrapBatchInputLayer(void);

		// Take each image in Mat, convert it to the correct image type,
		// color depth, size to match the net input. Convert to 
		// F32 type, since that's what the net inputs are. 
		// Subtract out the mean before passing to the net input
		// Then actually write the images to the net input memory buffers
		void PreprocessBatch(const std::vector< cv::Mat > &imgs);

		// Get the output values for a set of images
		// These values will be in the same order as the labels for each
		// image, and each set of labels for an image next adjacent to the
		// one for the next image.
		// That is, [0] = value for label 0 for the first image up to 
		// [n] = value for label n for the first image. It then starts again
		// for the next image - [n+1] = label 0 for image #2.
		std::vector<float> PredictBatch(const std::vector< cv::Mat > &imgs);

	private:
		// TODO : try shared pointers
		nvinfer1::IRuntime* runtime_;          // network runtime
		nvinfer1::ICudaEngine* engine_;        // network engine
		nvinfer1::IExecutionContext *context_; // netowrk context to run on engine
		cudaStream_t stream_;
		void* buffers_[2];           // input and output GPU buffers
		int inputIndex_;
		int outputIndex_;
		int numChannels_;

		float *inputCPU_;            // input CPU buffer
		std::vector< std::vector<cv::Mat> > inputBatch_; // net input buffers wrapped in Mats
};
