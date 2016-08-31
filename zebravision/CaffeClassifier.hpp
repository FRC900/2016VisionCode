#pragma once
#include <caffe/caffe.hpp>
#include "Classifier.hpp"

// CPU and GPU code is basically the same, so make the matrix
// type used a template parameter.
// For a CPU classifier, use CaffeClassifer<cv::Mat> fooCPU, and
// use CaffeClassifier<cv::gpu::GpuMat> fooGPU
template <class MatT>
class CaffeClassifier : public Classifier
{
	public:
		CaffeClassifier(const std::string& modelFile,
					const std::string& trainedFile,
					const std::string& zcaWeightFile,
					const std::string& labelFile,
					const size_t batchSize);
		~CaffeClassifier();

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
		void PreprocessBatch(const std::vector< MatT > &imgs);

		// Get the output values for a set of images
		// These values will be in the same order as the labels for each
		// image, and each set of labels for an image next adjacent to the
		// one for the next image.
		// That is, [0] = value for label 0 for the first image up to 
		// [n] = value for label n for the first image. It then starts again
		// for the next image - [n+1] = label 0 for image #2.
		std::vector<float> PredictBatch(const std::vector< MatT > &imgs);

		// Method which returns either mutable_cpu_data or mutable_gpu_data
		// depending on whether we're using CPU or GPU Mats
		// X will always be floats for our net, but C++ syntax
		// makes us make it a generic prototype here?
		template <class X>
			float *GetBlobData(caffe::Blob<X> *blob);

		// Method specialized to return either true or false depending
		// on whether we're using GpuMats or Mats
		bool IsGPU(void) const;

		std::shared_ptr<caffe::Net<float> > net_; // the net itself
		std::vector< std::vector<MatT> > inputBatch_; // net input buffers wrapped in Mats
};
