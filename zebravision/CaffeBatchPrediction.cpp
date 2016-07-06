#include <iostream>
#include <sys/stat.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "CaffeBatchPrediction.hpp"

using namespace std;

static bool fileExists(const char *filename)
{
	struct stat statBuffer;
	return stat(filename, &statBuffer) == 0;
}

template <class MatT>
CaffeClassifier<MatT>::CaffeClassifier(const string& modelFile,
      const string& trainedFile,
      const string& meanFile,
      const string& labelFile,
      const int batchSize) 
{
	if (!fileExists(modelFile.c_str()))
	{
		cerr << "Could not find Caffe model " << modelFile << endl;
		return;
	}
	if (!fileExists(trainedFile.c_str()))
	{
		cerr << "Could not find Caffe trained weights " << trainedFile << endl;
		return;
	}
	if (!fileExists(meanFile.c_str()))
	{
		cerr << "Could not find mean image file " << meanFile << endl;
		return;
	}
	if (!fileExists(labelFile.c_str()))
	{
		cerr << "Could not find label file " << labelFile << endl;
		return;
	}
	cout << "Loading " << modelFile << " " << trainedFile << " "<< meanFile << " " << labelFile << endl;

	if (IsGPU())
		caffe::Caffe::set_mode(caffe::Caffe::GPU);
	else
		caffe::Caffe::set_mode(caffe::Caffe::CPU);

	/* Set batchsize */
	batchSize_ = batchSize;

	/* Load the network - this includes model geometry and trained weights */
	net_.reset(new caffe::Net<float>(modelFile, caffe::TEST));
	net_->CopyTrainedLayersFrom(trainedFile);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	caffe::Blob<float>* inputLayer = net_->input_blobs()[0];
	numChannels_ = inputLayer->channels();
	CHECK(numChannels_ == 3 || numChannels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	inputGeometry_ = cv::Size(inputLayer->width(), inputLayer->height());

	/* Load the binaryproto mean file. */
	SetMean(meanFile);

	/* Load labels. */
	// This will be used to give each index of the output
	// a human-readable name
	ifstream labels(labelFile.c_str());
	CHECK(labels) << "Unable to open labels file " << labelFile;
	string line;
	while (getline(labels, line))
		labels_.push_back(string(line));

	caffe::Blob<float>* outputLayer = net_->output_blobs()[0];
	CHECK_EQ(labels_.size(), outputLayer->channels())
		<< "Number of labels is different from the output layer dimension.";

	// Pre-process Mat wrapping
	inputLayer->Reshape(batchSize_, numChannels_,
			inputGeometry_.height,
			inputGeometry_.width);

	/* Forward dimension change to all layers. */
	net_->Reshape();

	// The wrap code puts the buffer for one individual channel
	// input to the net (one color channel of one image) into 
	// a separate Mat 
	// The inner vector here will be one Mat per channel of the 
	// input to the net. The outer vector is a vector of those
	// one for each of the batched inputs.
	// This allows an easy copy from the input images
	// into the input buffers for the net
	WrapBatchInputLayer();
}

// Helper function for compare - used to sort values by pair.first keys
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
template <class MatT>
vector< vector<Prediction> > CaffeClassifier<MatT>::ClassifyBatch(
		const vector< MatT > &imgs, 
		size_t numClasses)
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

/* Load the mean file in binaryproto format. */
template <class MatT>
void CaffeClassifier<MatT>::SetMean(const string& meanFile) 
{
	caffe::BlobProto blobProto;
	ReadProtoFromBinaryFileOrDie(meanFile.c_str(), &blobProto);

	/* Convert from BlobProto to Blob<float> */
	caffe::Blob<float> meanBlob;
	meanBlob.FromProto(blobProto);
	CHECK_EQ(meanBlob.channels(), numChannels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	vector<cv::Mat> channels;
	float* data = meanBlob.mutable_cpu_data();
	for (int i = 0; i < numChannels_; ++i) 
	{
		/* Extract an individual channel. */
		cv::Mat channel(meanBlob.height(), meanBlob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += meanBlob.height() * meanBlob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	 * filled with this value. */
	cv::Scalar channelMean = cv::mean(mean);

	// Hack to possibly convert to GpuMat - if MatT is GpuMat,
	// this will upload the Mat object to a GpuMat, otherwise
	// it will just copy it to the member variable mean_
	mean_ = MatT(inputGeometry_, mean.type(), channelMean);
}

// TODO : see if we can do this once at startup or if
// it has to be done each pass.  If it can be done once,
// we can wrap the nets in Mat arrays in the constructor 
// and re-use them multiple times?
template <class MatT>
void CaffeClassifier<MatT>::setBatchSize(size_t batchSize) 
{
	CHECK(batchSize >= 0);
	if (batchSize == batchSize_) return;
	batchSize_ = batchSize;
	reshapeNet();
}

template <class MatT>
void CaffeClassifier<MatT>::reshapeNet() 
{
	CHECK(net_->input_blobs().size() == 1);
	caffe::Blob<float>* inputLayer = net_->input_blobs()[0];
	inputLayer->Reshape(batchSize_, numChannels_,
						 inputGeometry_.height,
						 inputGeometry_.width);
	net_->Reshape();
	WrapBatchInputLayer();
}

// Get the output values for a set of images in one flat vector
// These values will be in the same order as the labels for each
// image, and each set of labels for an image next adjacent to the
// one for the next image.
// That is, [0] = value for label 0 for the first image up to 
// [n] = value for label n for the first image. It then starts again
// for the next image - [n+1] = label 0 for image #2.
template <class MatT>
vector<float> CaffeClassifier<MatT>::PredictBatch(
		const vector<MatT> &imgs) 
{
	// Process each image so they match the format
	// expected by the net, then copy the images
	// into the net's input buffers
	PreprocessBatch(imgs);
	// Run a forward pass with the data filled in from above
	net_->ForwardPrefilled();
	/* Copy the output layer to a flat vector */
	caffe::Blob<float>* outputLayer = net_->output_blobs()[0];
	const float* begin = outputLayer->cpu_data();
	const float* end = begin + outputLayer->channels()*imgs.size();
	return vector<float>(begin, end);
}

// Wrap input layer of the net into separate Mat objects
// This sets them up to be written with actual data
// in PreprocessBatch()
// TODO : handle 
template <class MatT>
void CaffeClassifier<MatT>::WrapBatchInputLayer(void)
{
	caffe::Blob<float>* inputLayer = net_->input_blobs()[0];

	int width = inputLayer->width();
	int height = inputLayer->height();
	int num = inputLayer->num();
	float* inputData = GetBlobData(inputLayer);
	inputBatch_.clear();
	for ( int j = 0; j < num; j++)
	{
		vector<MatT> inputChannels;
		for (int i = 0; i < inputLayer->channels(); ++i)
		{
			MatT channel(height, width, CV_32FC1, inputData);
			inputChannels.push_back(channel);
			inputData += width * height;
		}
		inputBatch_.push_back(vector<MatT>(inputChannels));
	}
}

// Slow path for Preprocess - used if the image has to be
// color converted, resized, or resampled to float
template <class MatT>
void CaffeClassifier<MatT>::SlowPreprocess(const MatT &img, MatT &output)
{
	/* Convert the input image to the input image format of the network. */
	MatT sample;
	if (img.channels() == 3 && numChannels_ == 1)
		cvtColor(img, sample, CV_BGR2GRAY);
	else if (img.channels() == 4 && numChannels_ == 1)
		cvtColor(img, sample, CV_BGRA2GRAY);
	else if (img.channels() == 4 && numChannels_ == 3)
		cvtColor(img, sample, CV_BGRA2BGR);
	else if (img.channels() == 1 && numChannels_ == 3)
		cvtColor(img, sample, CV_GRAY2BGR);
	else
		sample = img;

	MatT sampleResized;
	if (sample.size() != inputGeometry_)
		resize(sample, sampleResized, inputGeometry_);
	else
		sampleResized = sample;

	MatT sampleFloat;
	if (((numChannels_ == 3) && (sampleResized.type() == CV_32FC3)) ||
		((numChannels_ == 1) && (sampleResized.type() == CV_32FC1)) )
		sampleFloat = sampleResized;
	else if (numChannels_ == 3)
		sampleResized.convertTo(sampleFloat, CV_32FC3);
	else
		sampleResized.convertTo(sampleFloat, CV_32FC1);

	subtract(sampleFloat, mean_, output);
}

// Take each image in Mat, convert it to the correct image type,
// color depth, size to match the net input. Convert to 
// F32 type, since that's what the net inputs are. 
// Subtract out the mean before passing to the net input
// Then actually write the images to the net input memory buffers
template <class MatT>
void CaffeClassifier<MatT>::PreprocessBatch(const vector<MatT> &imgs)
{
	CHECK(imgs.size() <= batchSize_) <<
		"PreprocessBatch() : too many input images : batch size is " << batchSize_ << "imgs.size() = " << imgs.size(); 

	for (size_t i = 0 ; i < imgs.size(); i++)
	{
		// If image is already the correct format,
		// don't both resizing/converting it again
		if ((imgs[i].channels() != numChannels_) ||
			(imgs[i].size()     != inputGeometry_) ||
			((numChannels_ == 3) && (imgs[i].type() != CV_32FC3)) ||
			((numChannels_ == 1) && (imgs[i].type() != CV_32FC1)) )
			SlowPreprocess(imgs[i], sampleNormalized_);
		else
			subtract(imgs[i], mean_, sampleNormalized_);

		/* This operation will write the separate BGR planes directly to the
		 * input layer of the network because it is wrapped by the MatT
		 * objects in inputChannels. */
		vector<MatT> *inputChannels = &inputBatch_.at(i);
		split(sampleNormalized_, *inputChannels);

#if 1
		// TODO : CPU Mats + GPU Caffe fails if this isn't here, no idea why
		if (i == 0)
			CHECK(reinterpret_cast<float*>(inputChannels->at(0).data)
					== GetBlobData(net_->input_blobs()[0]))
				<< "Input channels are not wrapping the input layer of the network.";
#endif
	}
}

// Assorted helper functions
template <class MatT>
size_t CaffeClassifier<MatT>::BatchSize(void) const
{
	return batchSize_;
}

template <class MatT>
cv::Size CaffeClassifier<MatT>::getInputGeometry(void) const
{
	return inputGeometry_;
}

template <class MatT>
const MatT CaffeClassifier<MatT>::getMean(void) const
{
	return mean_;
}

// TODO : maybe don't specialize this one in case
// we use something other than Mat or GpuMat in the
// future?
template <> template <>
float *CaffeClassifier<cv::Mat>::GetBlobData(caffe::Blob<float> *blob)
{
	return blob->mutable_cpu_data();
}

template <> template <>
float *CaffeClassifier<cv::gpu::GpuMat>::GetBlobData(caffe::Blob<float> *blob)
{
	return blob->mutable_gpu_data();
}

// TODO : maybe don't specialize this one in case
// we use something other than Mat or GpuMat in the
// future?
template <>
bool CaffeClassifier<cv::Mat>::IsGPU(void) const
{
	return true;
}

template <>
bool CaffeClassifier<cv::gpu::GpuMat>::IsGPU(void) const
{
	return true;
}

template class CaffeClassifier<cv::Mat>;
template class CaffeClassifier<cv::gpu::GpuMat>;
