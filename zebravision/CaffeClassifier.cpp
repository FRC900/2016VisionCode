#include <iostream>
#include <string>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "CaffeClassifier.hpp"
#include "Utilities.hpp"

using namespace std;
using namespace caffe;
using namespace cv;
using namespace cv::gpu;

// Google logging init stuff needs to happen
// just once per program run.  Use this
// var to make sure it does.
static bool glogInit_ = false;

template <class MatT>
CaffeClassifierThread<MatT>::CaffeClassifierThread(const string &modelFile,
							const string &trainedFile,
							const ZCA    &zca,      
							const size_t batchSize,
		std::shared_ptr<SynchronizedQueue<InQData<MatT>>> inQ,
		std::shared_ptr<SynchronizedQueue<OutQData>>      outQ) :
	initialized_(false),
	zca_(zca),
	inQ_(inQ), // input and output queues for communicating
	outQ_(outQ) // with the master thread
{
	// Make sure the model definition and 
	// weight files exist
	if (!utils::fileExists(modelFile))
	{
		cerr << "Could not find Caffe model " << modelFile << endl;
		return;
	}
	if (!utils::fileExists(trainedFile))
	{
		cerr << "Could not find Caffe trained weights " << trainedFile << endl;
		return;
	}
	cout << "Loading Caffe worker thread " << endl << "\t" << modelFile << endl << "\t" << trainedFile << endl;

	// Hopefully this turns off any logging
	// Run only once the first time and CaffeClassifer
	// class is created
	if (!glogInit_)
	{
		::google::InitGoogleLogging("");
		::google::LogToStderr();
		::google::SetStderrLogging(3);
		glogInit_ = true;
	}

	// Switch to CPU or GPU mode depending on
	// which version of the class we're running
	Caffe::set_mode(IsGPU() ? Caffe::GPU : Caffe::CPU);

	// Load the network - this includes model 
	// geometry and trained weights
	net_.reset(new Net<float>(modelFile, TEST));
	net_->CopyTrainedLayersFrom(trainedFile);

	// Some basic checking to make sure life makes sense
	// Protip - it never really does
	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	// More sanity checking. Number of input channels
	// should be 3 since we're using color images.
	Blob<float>* inputLayer = net_->input_blobs()[0];
	const int numChannels = inputLayer->channels();
	CHECK(numChannels == 3) << "Input layer should have 1 or 3 channels.";

	// Also, make sure the input geometry matches
	// the size expected by the preprocessing filters
	if (zca_.size() != Size(inputLayer->width(), inputLayer->height()))
	{
		cerr << "Net size != ZCA size" << endl;
		return;
	}

#if 0
	// Quick check to make sure there are enough labels
	// for each output
	Blob<float>* outputLayer = net_->output_blobs()[0];
	CHECK_EQ(this->labels_.size(), outputLayer->channels())
		<< "Number of labels is different from the output layer dimension.";
#endif

	// Set the network up for the specified batch
	// size. This processes a number of images in 
	// parallel. This works out to be a bit quicker
	// than doing them one by one
	inputLayer->Reshape(batchSize, numChannels,
						inputLayer->height(),
						inputLayer->width());

	// Forward dimension change to all layers
	net_->Reshape();

	// The wrap code puts the buffer for each individual channel
	// input to the net (one color channel of one image) into 
	// a separate Mat 
	// The inner vector here will be one Mat per channel of the 
	// input to the net. The outer vector is a vector of those
	// one for each of the batched input images.
	// This allows an easy copy from the input images
	// into the input buffers for the net by simply doing
	// an OpenCV split() call to split a 3-channel input
	// image into 3 separate 1-channel images arranged in
	// the correct order
	// 
	// GPU code path writes directly to the mutable_gpu_data()
	// pointer so no wrapping is needed
	if (!IsGPU())
	{
		Blob<float>* inputLayer = net_->input_blobs()[0];

		const size_t width  = inputLayer->width();
		const size_t height = inputLayer->height();
		const size_t num    = inputLayer->num();

		float* inputData = inputLayer->mutable_cpu_data();

		inputBatch_.clear();

		for (size_t j = 0; j < num; j++)
		{
			vector<MatT> inputChannels;
			for (size_t i = 0; i < inputLayer->channels(); ++i)
			{
				MatT channel(height, width, CV_32FC1, inputData);
				inputChannels.push_back(channel);
				inputData += width * height;
			}
			inputBatch_.push_back(vector<MatT>(inputChannels));
		}
	}
	initialized_ = true;
}
// Get the output values for a set of images in one flat vector
// These values will be in the same order as the labels for each
// image, and each set of labels for an image next adjacent to the
// one for the next image.
// That is, [0] = value for label 0 for the first image up to 
// [n] = value for label n for the first image. It then starts again
// for the next image - [n+1] = label 0 for image #2.
template <class MatT>
void CaffeClassifierThread<MatT>::operator() ()
{
	// Loop forever waiting for data to process
	while (true)
	{
		// Get input images from the queue, blocking if none
		// is available
		InQData<MatT> input = inQ_->Dequeue();

		// Convert each image so they match the format
		// expected by the net, then copy the images
		// into the net's input buffers
		PreprocessBatch(input.data_);

		// Run a forward pass with the data filled in from above
		// This is the step which actually runs the net over
		// the input images. It produces a set of confidence scores 
		// per image - one score per label.  The higher
		// the score, the more likely the net thinks
		// it is that the input image matches a given
		// label.
		net_->Forward();

		// Copy the output layer to a flat vector
		Blob<float>* outputLayer = net_->output_blobs()[0];
		const float* begin = outputLayer->cpu_data();
		const float* end = begin + outputLayer->channels() * input.data_.size();
		outQ_->Enqueue(OutQData(input.batchNum_, vector<float>(begin, end)));
		// Make sure we can be interrupted
		boost::this_thread::interruption_point();
	}
}

// Take each image in Mat, convert it to the correct image size,
// and apply ZCA whitening to preprocess the files
// Then actually write the images to the net input memory buffers
template <>
void CaffeClassifierThread<Mat>::PreprocessBatch(const vector<Mat> &imgs)
{
	vector<Mat> zcaImgs = this->zca_.Transform32FC3(imgs);

	// Hack to reset input layer to think that
	// data is on the CPU side.  Only really needed
	// when CPU & GPU operations are combined
	net_->input_blobs()[0]->mutable_cpu_data()[0] = 0;

	// For each image in the list, copy it to 
	// the net's input buffer
	for (size_t i = 0 ; i < zcaImgs.size(); i++)
	{
		/* This operation will write the separate BGR planes directly to the
		 * input layer of the network because it is wrapped by the MatT
		 * objects in inputChannels. */
		vector<Mat> *inputChannels = &inputBatch_.at(i);
		split(zcaImgs[i], *inputChannels);
	}
}

// Take each image in GpuMat, convert it to the correct image type,
// and apply ZCA whitening to preprocess the files
// The GPU input to the net is passed in to Transform32FC3 and
// that function copies its final results directly into the
// input buffers of the net.
template <>
void CaffeClassifierThread<GpuMat>::PreprocessBatch(const vector<GpuMat> &imgs)
{
	float* inputData = net_->input_blobs()[0]->mutable_gpu_data();
	this->zca_.Transform32FC3(imgs, inputData);
}


// Specialize these functions - the Mat one works
// on the CPU while the GpuMat one works on the GPU
template <>
bool CaffeClassifierThread<Mat>::IsGPU(void) const
{
	// TODO : change to unconditional false
	// eventually once things are debugged
	//return (getCudaEnabledDeviceCount() > 0);
	return false;
}


template <>
bool CaffeClassifierThread<GpuMat>::IsGPU(void) const
{
	return true;
}


template <class MatT>
bool CaffeClassifierThread<MatT>::initialized(void) const
{
	return initialized_;
}


// Instantiate both Mat and GpuMat versions of the Classifier
template class CaffeClassifierThread<Mat>;
template class CaffeClassifierThread<GpuMat>;
