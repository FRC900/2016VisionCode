#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "Classifier.hpp"

using namespace std;
using namespace cv;

// Move to base class method?
bool Classifier::fileExists(const string &fileName) const
{
	struct stat statBuffer;
	return stat(fileName.c_str(), &statBuffer) == 0;
}

Classifier::Classifier(const string& modelFile,
      const string& trainedFile,
      const string& zcaWeightFile,
      const string& labelFile,
      const size_t  batchSize) :
	batchSize_(batchSize),
	zca_(zcaWeightFile.c_str())
{
	(void)modelFile;
	(void)trainedFile;
	if (!fileExists(zcaWeightFile))
	{
		cerr << "Could not find ZCA weight file " << zcaWeightFile << endl;
		return;
	}
	if (!fileExists(labelFile))
	{
		cerr << "Could not find label file " << labelFile << endl;
		return;
	}

	// Load labels
	// This will be used to give each index of the output
	// a human-readable name
	ifstream labels(labelFile.c_str());
	if (!labels) 
		cerr << "Unable to open labels file " << labelFile << endl;
	string line;
	while (getline(labels, line))
		labels_.push_back(string(line));

	inputGeometry_ = zca_.size();
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
vector< vector<Prediction> > Classifier::ClassifyBatch(
		const vector<Mat> &imgs, size_t numClasses)
{
	// outputBatch will be a flat vector of N floating point values 
	// per image (1 per N output labels), repeated
	// times the number of input images batched per run
	// Convert that into the output vector of vectors
	vector<float> outputBatch = PredictBatch(imgs);
	vector< vector<Prediction> > predictions;
	size_t labelsSize = labels_.size();
	numClasses = min(numClasses, labelsSize);
	// For each image, find the top numClasses values
	for(size_t j = 0; j < imgs.size(); j++)
	{
		// Create an output vector just for values for this image. Since
		// each image has labelsSize values, that's outputBatch[j*labelsSize]
		// through outputBatch[(j+1) * labelsSize]
		vector<float> output(outputBatch.begin() + j*labelsSize, outputBatch.begin() + (j+1)*labelsSize);
		// For the output specific to the jth image, grab the
		// indexes of the top numClasses predictions
		vector<int> maxN = Argmax(output, numClasses);
		// Using those top N indexes, create a set of labels/value predictions
		// specific to this jth image
		vector<Prediction> predictionSingle;
		for (size_t i = 0; i < numClasses; ++i) 
		{
			int idx = maxN[i];
			predictionSingle.push_back(make_pair(labels_[idx], output[idx]));
		}
		// Add the predictions for this image to the list of
		// predictions for all images
		predictions.push_back(vector<Prediction>(predictionSingle));
	}
	return predictions;
}

// Get the output values for a set of images in one flat vector
// These values will be in the same order as the labels for each
// image, and each set of labels for an image next adjacent to the
// one for the next image.
// That is, [0] = value for label 0 for the first image up to 
// [n] = value for label n for the first image. It then starts again
// for the next image - [n+1] = label 0 for image #2.
vector<float> Classifier::PredictBatch(const vector<Mat> &imgs) 
{
	// Should be implemented in derived classes
	(void) imgs;
	return vector<float>();
}

// Assorted helper functions
size_t Classifier::BatchSize(void) const
{
	return batchSize_;
}

Size Classifier::getInputGeometry(void) const
{
	return inputGeometry_;
}

