#pragma once
#include <memory>
#include <string>
#include <vector>
#include <queue>
#include <utility>

#include <boost/thread.hpp>

#include <opencv2/core/core.hpp>

#include "zca.hpp"

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<std::string, float> Prediction;
// Queue used to pass work into a worker thread.
// Holds batchNumber - where the data came from
// in the original input, and input
// image data vector
template <typename MatT>
class InQData
{
	public:
		int batchNum_;
		std::vector<MatT> data_;
};

// Output data coming from worker queue back
// to main thread. Holds batch number and sets
// of confidence values for each input image
class OutQData
{
	public:
		OutQData(void)
		{
		}
		OutQData(int batchNum, const std::vector<float> &data) :
			batchNum_(batchNum),
			data_(data)
		{
		}

		int batchNum_;
		std::vector<float> data_;
};

// From https://www.quantnet.com/threads/c-multithreading-in-boost.10028/
// Queue class that has thread synchronisation
template <typename T>
class SynchronizedQueue
{
	public:
		SynchronizedQueue(void) : 
			exit_(false)
		{
		}
		// Add data to the queue and notify others
		void Enqueue(const T& data)
		{
			// Acquire lock on the queue
			boost::unique_lock<boost::mutex> lock(mutex_);

			// Add the data to the queue
			queue_.push(data);

			// Notify others that data is ready
			cond_.notify_one();
		} // Lock is automatically released here

		// Get data from the queue. Wait for data if not available
		T Dequeue(void)
		{
			// Acquire lock on the queue
			boost::unique_lock<boost::mutex> lock(mutex_);

			// When there is no data, wait till someone fills it.
			// Lock is automatically released in the wait and obtained
			// again after the wait
			while (queue_.size() == 0)
			{
				if (exit_)
					return T();
				cond_.wait(lock);
			}

			// Retrieve the data from the queue
			T result = queue_.front(); 
			queue_.pop();
			return result;
		} // Lock is automatically released here

		void signalExit(void)
		{
			exit_ = true;
			cond_.notify_all();
		}

		bool checkExit(void) const
		{
			return exit_;
		}

	private:
		std::queue<T> queue_; // Use STL queue to store data
		boost::mutex mutex_; // The mutex to synchronise on
		boost::condition_variable cond_; // The condition to wait for
		bool exit_;
};
 
template<class MatT, class ClassifierT>
class Classifier 
{
	public:
		Classifier(const std::string &modelFile,
				   const std::string &weightsFile,
				   const std::string &zcaWeightFile,
				   const std::string &labelFile,
				   const size_t       batchSize,
				   const size_t       numThreads);
		~Classifier();

		// Given X input images, return X vectors of predictions.
		// Each prediction is a label, value pair, where the value is
		// the confidence level for each given label.
		// Each of the X vectors are themselves a vector which will have the 
		// N predictions with the highest confidences for the corresponding
		// input image
		std::vector<std::vector<Prediction>> ClassifyBatch(const std::vector<MatT> &imgs, const size_t numClasses);

		// Get the width and height of an input image to the net
		cv::Size getInputGeometry(void) const;

		// Get the batch size of the model
		size_t batchSize(void) const;

		// See if the classifier loaded correctly
		bool initialized(void) const;

	protected:
		bool fileExists(const std::string &fileName) const;
		cv::Size inputGeometry_;          // size of one input image
		size_t batchSize_;                // number of images to process in one go
		ZCA  zca_;                        // weights used to normalize input data
		bool initialized_;
		std::vector<std::string> labels_; // labels for each output index
		std::shared_ptr<boost::thread_group> threads_;     // set of threads which process input data.  

		std::shared_ptr<SynchronizedQueue<InQData<MatT>>> inQ_;  // queues for communicating with
		std::shared_ptr<SynchronizedQueue<OutQData> >     outQ_; // worker threads

	private:
		// Grab prediction data from the output queue 
		// and put it into a vector of predictions.
		// Each image has numClasses predictions, and the
		// outer vector has one set of these per input
		// image
		std::vector<std::vector<Prediction>> processPredictions(const size_t imgSize, size_t batchCount, const size_t numClasses);
};
