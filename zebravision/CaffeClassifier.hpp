#pragma once
#include <caffe/caffe.hpp>
#include "Classifier.hpp"

// CPU and GPU code is basically the same, so make the matrix
// type used a template parameter.
// For a CPU classifier, use CaffeClassifer<cv::Mat> fooCPU, and
// use CaffeClassifierThread<cv::gpu::GpuMat> fooGPU
template <class MatT>
class CaffeClassifierThread
{
	public :
		CaffeClassifierThread(const std::string &modelFile,
							  const std::string &trainedFile,
							  const ZCA         &zca,
							  const size_t      batchSize,
				std::shared_ptr<SynchronizedQueue<InQData<MatT>>> inQ,
				std::shared_ptr<SynchronizedQueue<OutQData>>      outQ);

		// Get the output values for a set of images in one flat vector
		// These values will be in the same order as the labels for each
		// image, and each set of labels for an image next adjacent to the
		// one for the next image.
		// That is, [0] = value for label 0 for the first image up to 
		// [n] = value for label n for the first image. It then starts again
		// for the next image - [n+1] = label 0 for image #2.
		void operator() ();

		bool initialized(void) const;

	private:
		// Take each image in Mat, convert it to the correct image type,
		// color depth, size to match the net input. Convert to 
		// F32 type, since that's what the net inputs are. 
		// Subtract out the mean before passing to the net input
		// Then actually write the images to the net input memory buffers
		void PreprocessBatch(const std::vector<MatT> &imgs);
		// Method specialized to return either true or false depending
		// on whether we're using GpuMats or Mats
		bool IsGPU(void) const;

		std::shared_ptr<caffe::Net<float> > net_; // the net itself
		std::vector< std::vector<MatT>> inputBatch_; // net input buffers wrapped in Mats - used for CPU code only 
		bool initialized_;   // set to true once the net is correctly initialzied
		ZCA zca_;
		
		std::shared_ptr<SynchronizedQueue<InQData<MatT>>> inQ_; // input and output queues for communicating
		std::shared_ptr<SynchronizedQueue<OutQData>> outQ_;       // data into and out of the thread
};

