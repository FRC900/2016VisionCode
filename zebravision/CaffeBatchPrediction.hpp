#ifndef INC__CAFFEBATCHPREDICTION_HPP_
#define INC__CAFFEBATCHPREDICTION_HPP_

#include <memory>
#include <queue>
#include <string>
#include <vector>
#include <utility>

#include <boost/thread.hpp>

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<std::string, float> Prediction;

// Data passed into and out of worker threads
template <typename MatT> class InQData;
class OutQData;

// From https://www.quantnet.com/threads/c-multithreading-in-boost.10028/
// Queue class that has thread synchronisation. Used
// for thread-safe communication between main thread
// and workers
template <typename T> class SynchronizedQueue;

// CPU and GPU code is basically the same, so make the input 
// matrix type used a template parameter.
// For a CPU classifier, use CaffeClassifer<cv::Mat> fooCPU, and
// use CaffeClassifier<cv::gpu::GpuMat> fooGPU
template <class MatT>
class CaffeClassifier 
{
	public:
		CaffeClassifier(const std::string& modelFile,
						const std::string& trainedFile,
						const std::string& meanFile,
						const std::string& labelFile,
						int batchSize,
						int numThreads = 4);

		// Given X input images, return X vectors of predictions.
		// Each prediction is a label, value pair, where the value is
		// the confidence level for each given label.
		// Each of the X vectors are themselves a vector which will have the 
		// N predictions with the highest confidences for the corresponding
		// input image
		std::vector< std::vector<Prediction> > ClassifyBatch(const std::vector< MatT > &imgs, size_t numClasses);

		// Get the width and height of an input image to the net
		cv::Size getInputGeometry(void) const;

		// Get the batch size of the model
		size_t BatchSize(void) const;

	private:
		// Method specialized to return either true or false depending
		// on whether we're using GpuMats or Mats
		bool IsGPU(void) const;

		size_t batchSize_;                // number of images to process in one go
		cv::Size inputGeometry_;          // size of one input image
		std::vector<std::string> labels_; // labels for each output value

		std::shared_ptr<boost::thread_group> threads_;     // set of threads which process input data.  

		std::shared_ptr<SynchronizedQueue<InQData<MatT>>> inQ_;  // queues for communicating with
		std::shared_ptr<SynchronizedQueue<OutQData> >     outQ_; // worker threads
};

#endif
