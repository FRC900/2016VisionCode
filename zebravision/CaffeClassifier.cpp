#include <iostream>
#include <sys/stat.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "CaffeClassifier.hpp"

using namespace std;
using namespace caffe;
using namespace cv;
using namespace cv::gpu;

static bool glogInit_ = false;

#if 0
#include <sys/time.h>
static double gtod_wrapper(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#endif

template <class MatT>
CaffeClassifier<MatT>::CaffeClassifier(const string& modelFile,
      const string& trainedFile,
      const string& zcaWeightFile,
      const string& labelFile,
      const size_t  batchSize) :
	Classifier<MatT>(modelFile, trainedFile, zcaWeightFile, labelFile, batchSize),
	initialized_(false)
{
	if (!Classifier<MatT>::initialized())
		return;
	if (!this->fileExists(modelFile))
	{
		cerr << "Could not find Caffe model " << modelFile << endl;
		return;
	}
	if (!this->fileExists(trainedFile))
	{
		cerr << "Could not find Caffe trained weights " << trainedFile << endl;
		return;
	}
	cout << "Loading Caffe model " << modelFile << " " << trainedFile << " " << zcaWeightFile << " " << labelFile << endl;

	if (IsGPU())
		Caffe::set_mode(Caffe::GPU);
	else
		Caffe::set_mode(Caffe::CPU);

	// Hopefully this turns off any logging
	if (!glogInit_)
	{
		::google::InitGoogleLogging("");
		::google::LogToStderr();
		::google::SetStderrLogging(3);
		glogInit_ = true;
	}

	/* Load the network - this includes model geometry and trained weights */
	net_.reset(new Net<float>(modelFile, TEST));
	net_->CopyTrainedLayersFrom(trainedFile);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* inputLayer = net_->input_blobs()[0];
	const int numChannels = inputLayer->channels();
	CHECK(numChannels == 3 || numChannels == 1) << "Input layer should have 1 or 3 channels.";
	if (this->inputGeometry_ != Size(inputLayer->width(), inputLayer->height()))
	{
		cerr << "Net size != ZCA size" << endl;
		return;
	}

	Blob<float>* outputLayer = net_->output_blobs()[0];
	CHECK_EQ(this->labels_.size(), outputLayer->channels())
		<< "Number of labels is different from the output layer dimension.";

	// Pre-process Mat wrapping
	inputLayer->Reshape(batchSize, numChannels,
			inputLayer->height(),
			inputLayer->width());

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
	initialized_ = true;
}


template <class MatT>
CaffeClassifier<MatT>::~CaffeClassifier()
{
}

// Wrap input layer of the net into separate Mat objects
// This sets them up to be written with actual data
// in PreprocessBatch()
template <class MatT>
void CaffeClassifier<MatT>::WrapBatchInputLayer(void)
{
	Blob<float>* inputLayer = net_->input_blobs()[0];

	const int width  = inputLayer->width();
	const int height = inputLayer->height();
	const int num    = inputLayer->num();
	float* inputData = GetBlobData(inputLayer);

	inputBatch_.clear();

	for (int j = 0; j < num; j++)
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

// Get the output values for a set of images in one flat vector
// These values will be in the same order as the labels for each
// image, and each set of labels for an image next adjacent to the
// one for the next image.
// That is, [0] = value for label 0 for the first image up to 
// [n] = value for label n for the first image. It then starts again
// for the next image - [n+1] = label 0 for image #2.
template <class MatT>
vector<float> CaffeClassifier<MatT>::PredictBatch(const vector<MatT> &imgs) 
{
	// Process each image so they match the format
	// expected by the net, then copy the images
	// into the net's input buffers
	//double start = gtod_wrapper();
	PreprocessBatch(imgs);
	//cout << "PreprocessBatch " << gtod_wrapper() - start << endl;
	//start = gtod_wrapper();
	// Run a forward pass with the data filled in from above
	net_->Forward();
	//cout << "Forward " << gtod_wrapper() - start << endl;

	//start = gtod_wrapper();
	/* Copy the output layer to a flat vector */
	Blob<float>* outputLayer = net_->output_blobs()[0];
	const float* begin = outputLayer->cpu_data();
	const float* end = begin + outputLayer->channels()*imgs.size();
	//cout << "Output " << gtod_wrapper() - start << endl;
	return vector<float>(begin, end);
}

// TODO : maybe don't specialize this one in case
// we use something other than Mat or GpuMat in the
// future?
template <> template <>
float *CaffeClassifier<Mat>::GetBlobData(Blob<float> *blob)
{
	return blob->mutable_cpu_data();
}

template <> template <>
float *CaffeClassifier<GpuMat>::GetBlobData(Blob<float> *blob)
{
	return blob->mutable_gpu_data();
}

// Take each image in Mat, convert it to the correct image type,
// Then actually write the images to the net input memory buffers
template <>
void CaffeClassifier<Mat>::PreprocessBatch(const vector<Mat> &imgs)
{
	CHECK(imgs.size() <= this->batchSize_) <<
		"PreprocessBatch() : too many input images : batch size is " << this->batchSize_ << "imgs.size() = " << imgs.size(); 

#if 0
	for (size_t i = 0 ; i < imgs.size(); i++)
	{
		Mat img = imgs[i].clone();
		Mat wr;
		img.convertTo(wr, CV_8UC3, 255);
		stringstream s;
		s << "debug_ppb_before_" << i << ".png";
		imwrite(s.str(), wr);
	}
#endif
	vector<Mat> zcaImgs = this->zca_.Transform32FC3(imgs);
#if 0
	for (size_t i = 0 ; i < zcaImgs.size(); i++)
	{
		Mat img = zcaImgs[i].clone();
		Mat wr;
		img.convertTo(wr, CV_8UC3, 255, 127);
		stringstream s;
		s << "debug_ppb_after_" << i << ".png";
		imwrite(s.str(), wr);
	}
#endif

#if 0
	for (size_t i = 0 ; i < zcaImgs.size(); i++)
	{
		zcaImgs[i] *= 255.;
		zcaImgs[i] += 127.;
	}
#endif
	for (size_t i = 0 ; i < zcaImgs.size(); i++)
	{
		/* This operation will write the separate BGR planes directly to the
		 * input layer of the network because it is wrapped by the MatT
		 * objects in inputChannels. */
		vector<Mat> *inputChannels = &inputBatch_.at(i);
		split(zcaImgs[i], *inputChannels);

#if 1
		// TODO : CPU Mats + GPU Caffe fails if this isn't here, no idea why
		if (i == 0)
			CHECK(reinterpret_cast<float*>(inputChannels->at(0).data)
					== GetBlobData(net_->input_blobs()[0]))
				<< "Input channels are not wrapping the input layer of the network.";
#endif
	}
}

// Take each image in GpuMat, convert it to the correct image type,
// Then actually write the images to the net input memory buffers
template <>
void CaffeClassifier<GpuMat>::PreprocessBatch(const vector<GpuMat> &imgs)
{
	CHECK(imgs.size() <= this->batchSize_) <<
		"PreprocessBatch() : too many input images : batch size is " << this->batchSize_ << "imgs.size() = " << imgs.size(); 

	Blob<float>* inputLayer = net_->input_blobs()[0];
	float* inputData = GetBlobData(inputLayer);
	this->zca_.Transform32FC3(imgs, inputData);
}


// TODO : maybe don't specialize this one in case
// we use something other than Mat or GpuMat in the
// future?
template <>
bool CaffeClassifier<Mat>::IsGPU(void) const
{
	return true;
}

template <>
bool CaffeClassifier<GpuMat>::IsGPU(void) const
{
	return true;
}

template <class MatT>
bool CaffeClassifier<MatT>::initialized(void) const
{
	if (!Classifier<MatT>::initialized())
		return false;
	
	return initialized_;
}


template class CaffeClassifier<Mat>;
template class CaffeClassifier<GpuMat>;
