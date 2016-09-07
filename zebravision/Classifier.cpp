#include <iostream>
#include <fstream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "Classifier.hpp"
#include "CaffeClassifier.hpp"
#include "GIEClassifier.hpp"
#include "Utilities.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

template <class MatT, class ClassifierT>
Classifier<MatT, ClassifierT>::Classifier(const string& modelFile,
							 const string& trainedFile,
							 const string& zcaWeightFile,
							 const string& labelFile,
						 	 const size_t  batchSize,
							 const size_t  numThreads) :
	batchSize_(batchSize),
	zca_(zcaWeightFile.c_str(), batchSize),
	initialized_(false)
{
	if (!utils::fileExists(zcaWeightFile))
	{
		cerr << "Could not find ZCA weight file " << zcaWeightFile << endl;
		return;
	}
	if (!utils::fileExists(labelFile))
	{
		cerr << "Could not find label file " << labelFile << endl;
		return;
	}

	// Load labels
	// This will be used to give each index of the output
	// a human-readable name
	ifstream labels(labelFile.c_str());
	if (!labels) 
	{
		cerr << "Unable to open labels file " << labelFile << endl;
		return;
	}
	string line;
	while (getline(labels, line))
		labels_.push_back(string(line));

	// Create queues used to communicate with
	// worker threads
	inQ_  = make_shared<SynchronizedQueue<InQData<MatT>>>();
	outQ_ = make_shared<SynchronizedQueue<OutQData>>();

	// Create requested number of worker threads
	threads_ = std::make_shared<boost::thread_group>();
	for (int i = 0; i < numThreads; i++)
	{
		ClassifierT t(modelFile, trainedFile, zca_, batchSize, inQ_, outQ_);
		
		if (!t.initialized())
			return;

		threads_->create_thread(t);
	}

	inputGeometry_ = zca_.size();
	initialized_ = true;
}

// Helper function for compare - used to sort values by pair.first keys
// TODO : redo as lambda
static bool PairCompare(const pair<float, int>& lhs, 
						const pair<float, int>& rhs) 
{
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static vector<int> Argmax(const vector<float>& v, size_t N) 
{
	vector<pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(make_pair(v[i], i));
	partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	vector<int> result;
	for (size_t i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

// Given X input images, return X vectors of predictions.
// Each of the X vectors are themselves a vector which will have the 
// N predictions with the highest confidences for the corresponding
// input image
template <class MatT, class ClassifierT>
vector<vector<Prediction>> Classifier<MatT, ClassifierT>::ClassifyBatch(
		const vector<MatT> &imgs, const size_t numClasses)
{
	// outputBatch will be a flat vector of N floating point values 
	// per image (1 per N output labels), repeated
	// times the number of input images batched per run
	// Convert that into the output vector of vectors
	//cerr << "ClassifyBatch Mat->Mat" << endl;
	//
	// Input array size can be any length, but
	// the worker threads can handle at most batchSize
	// images at a time. Copy up to batchSize_ images
	// into one queue entry and send it off to the
	// worker threads to handle.
	int batchCount = 0;
	InQData<MatT> batchInput;
    for (auto it = imgs.cbegin(); it != imgs.cend(); ++it)
    {
		batchInput.data_.push_back(*it);
		if ((batchInput.data_.size() == batchSize_) || ((it + 1) == imgs.cend()))
		{
			// Enqueue this image batch. The call will kick
			// off a worker thread if one is idle
			// If not, the data will wait in th queue until a
			// worker thread frees up to handleit
			// Keep track of the batch number - this is 
			// a way to figure out which subset of the
			// full input array each queue entry
			// corresponds to.
			batchInput.batchNum_ = batchCount++;
			inQ_->Enqueue(batchInput);

			// Clear out vector to start fresh with
			// the next image in the input list
			batchInput.data_.clear();
		}
    }
	return processPredictions(imgs.size(), batchCount, numClasses);
}

template <class MatT, class ClassifierT>
vector<vector<Prediction>> Classifier<MatT, ClassifierT>::processPredictions(const size_t imgSize, size_t batchCount, const size_t numClasses)
{
	// Create an array of empty predictions long
	// enough to hold the results for every image
	vector< vector<Prediction> > predictions(imgSize, vector<Prediction>());

	// The code is set up to grab only the top
	// N predictions. Make sure that N doesn't
	// get larger than the number of actual
	// known labels for the net
	const size_t labelsSize = labels_.size();
	const size_t classes = min(numClasses, labelsSize);
	while (batchCount--)
	{
		// outputBatch will be a flat 1D vector of N floating point values 
		// per image (1 per N output labels), repeated
		// times the number of input images batched per run
		// Convert that into the 2D output vector of vectors
		OutQData outputBatch = outQ_->Dequeue();

		// For each image, find the top numClasses values
		for(size_t j = 0; j < (outputBatch.data_.size() / labelsSize); j++)
		{
			// Create an output vector just for values for this image. Since
			// each image has labelsSize values, that's outputBatch[j*labelsSize]
			// through outputBatch[(j+1) * labelsSize]
			vector<float> output(outputBatch.data_.begin() + j*labelsSize, outputBatch.data_.begin() + (j+1)*labelsSize);
			// For the output specific to the jth image, grab the
			// indexes of the top numClasses predictions
			// That is, maxN[0] will hold the index of the
			// highest-scoring output label/score. maxN[1] will
			// be the next best scoring one, and so on.
			vector<int> maxN = Argmax(output, classes);
			// Using those top N indexes, create a set of labels/value predictions
			// specific to this jth image
			vector<Prediction> predictionSingle;
			for (size_t i = 0; i < classes; ++i) 
			{
				int idx = maxN[i];
				predictionSingle.push_back(make_pair(labels_[idx], output[idx]));
			}
			// Add the predictions for this image to the list of
			// predictions for all images. Put it in the correct location
			// for the given batch and image
			predictions[outputBatch.batchNum_ * batchSize_ + j] = vector<Prediction>(predictionSingle);
		}
	}
	return predictions;
}

// Assorted helper functions
template <class MatT, class ClassifierT>
size_t Classifier<MatT, ClassifierT>::batchSize(void) const
{
	return batchSize_;
}

template <class MatT, class ClassifierT>
Size Classifier<MatT, ClassifierT>::getInputGeometry(void) const
{
	return inputGeometry_;
}

template <class MatT, class ClassifierT>
bool Classifier<MatT, ClassifierT>::initialized(void) const
{
	if (labels_.size() == 0)
		return false;
	if (inputGeometry_ == Size())
		return false;
	return initialized_;
}

template class Classifier<Mat, CaffeClassifierThread<Mat>>;
template class Classifier<GpuMat, CaffeClassifierThread<GpuMat>>;
template class Classifier<Mat, GIEClassifierThread<Mat>>;
template class Classifier<GpuMat, GIEClassifierThread<GpuMat>>;
