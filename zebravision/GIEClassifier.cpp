#ifdef USE_GIE
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>

#include <opencv2/core/core.hpp>
#include "caffeParser.h"
#include "GIEClassifier.hpp"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace std;
using namespace cv;
using namespace cv::gpu;

#define CHECK_CUDA(status)								\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}

// stuff we know about the network and the caffe input/output blobs
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "softmax";

// Logger for GIE info/warning/errors
class Logger : public ILogger			
{
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;

void caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					 const std::string& modelFile,				// name for model 
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
					 std::stringstream &gieModelStream) // serialized version of model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	CaffeParser* parser = new CaffeParser;
	const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(),
															  modelFile.c_str(),
															  *network,
															  nvinfer1::DataType::kFLOAT);

	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine_
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	delete parser;

	// serialize the engine_, then close everything down
	engine->serialize(gieModelStream);
	engine->destroy();
	builder->destroy();

}
GIEClassifier::GIEClassifier(const string& modelFile,
      const string& trainedFile,
      const string& zcaWeightFile,
      const string& labelFile,
      const size_t  batchSize) :
	Classifier(modelFile, trainedFile, zcaWeightFile, labelFile, batchSize),
	numChannels_(3),
	inputCPU_(NULL),
	initialized_(false)
{
	if (!Classifier<Mat>::initialized())
		return;

	if (!fileExists(modelFile))
	{
		cerr << "Could not find Caffe model " << modelFile << endl;
		return;
	}
	if (!fileExists(trainedFile))
	{
		cerr << "Could not find Caffe trained weights " << trainedFile << endl;
		return;
	}
	cout << "Loading GIE model " << modelFile << " " << trainedFile << " " << zcaWeightFile << " " << labelFile << endl;

	batchSize_ = batchSize;

	std::stringstream gieModelStream;
	// TODO :: read from file if exists and is newer than modelFile and trainedFile
	caffeToGIEModel(modelFile, trainedFile, std::vector < std::string > { OUTPUT_BLOB_NAME }, batchSize_, gieModelStream);

	// Create runable version of model by
	// deserializing the engine 
	gieModelStream.seekg(0, gieModelStream.beg);

	runtime_ = createInferRuntime(gLogger);
	engine_  = runtime_->deserializeCudaEngine(gieModelStream);
	context_ = engine_->createExecutionContext();

	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine_.getNbBindings() == 2);

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	inputIndex_  = engine_->getBindingIndex(INPUT_BLOB_NAME); 
	outputIndex_ = engine_->getBindingIndex(OUTPUT_BLOB_NAME);

	// create GPU buffers and a stream
	CHECK_CUDA(cudaMalloc(&buffers_[inputIndex_], batchSize * numChannels_ * inputGeometry_.area() * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&buffers_[outputIndex_], batchSize * labels_.size() * sizeof(float)));

	CHECK_CUDA(cudaStreamCreate(&stream_));

	// Set up input buffers for net
	WrapBatchInputLayer();

	initialized_ = true;
}

GIEClassifier::~GIEClassifier()
{
	delete [] inputCPU_;
	// release the stream and the buffers
	cudaStreamDestroy(stream_);
	CHECK_CUDA(cudaFree(buffers_[inputIndex_]));
	CHECK_CUDA(cudaFree(buffers_[outputIndex_]));
	context_->destroy();
	engine_->destroy();
	runtime_->destroy();
}

GIEClassifier::initialized(void) const
{
	if (!Classifier<Mat>::initialized())
		return false;
	
	return initialized_;
}

// Wrap input layer of the net into separate Mat objects
// This sets them up to be written with actual data
// in PreprocessBatch()
void GIEClassifier::WrapBatchInputLayer(void)
{
	if (inputCPU_)
		delete [] inputCPU_;
	inputCPU_= new float[batchSize_ * numChannels_ * inputGeometry_.area()];
	float *inputCPU = inputCPU_;

	inputBatch_.clear();

	for (size_t j = 0; j < batchSize_; j++)
	{
		vector<Mat> inputChannels;
		for (int i = 0; i < numChannels_; ++i)
		{
			Mat channel(inputGeometry_.height, inputGeometry_.width, CV_32FC1, inputCPU);
			inputChannels.push_back(channel);
			inputCPU += inputGeometry_.area();
		}
		inputBatch_.push_back(vector<Mat>(inputChannels));
	}
}

// Take each image in Mat, convert it to the correct image type,
// Then actually write the images to the net input memory buffers
void GIEClassifier::PreprocessBatch(const vector<Mat> &imgs)
{
	if (imgs.size() > batchSize_) 
		cerr <<
		"PreprocessBatch() : too many input images : batch size is " << 
		batchSize_ << "imgs.size() = " << imgs.size() << endl; 

	vector<Mat> zcaImgs = zca_.Transform32FC3(imgs);
	for (size_t i = 0 ; i < zcaImgs.size(); i++)
	{
		/* This operation will write the separate BGR planes directly to the
		 * inputCPU_ array since it is wrapped by the MatT
		 * objects in inputChannels. */
		vector<Mat> *inputChannels = &inputBatch_.at(i);
		split(zcaImgs[i], *inputChannels);

#if 1
		// TODO : CPU Mats + GPU Caffe fails if this isn't here, no idea why
		if (i == 0)
			if (reinterpret_cast<float*>(inputChannels->at(0).data) != inputCPU_)
				cerr << "Input channels are not wrapping the input layer of the network." << endl;
#endif
	}
}

vector<float> GIEClassifier::PredictBatch(const vector<Mat> &imgs)
{
	PreprocessBatch(imgs);
	float output[labels_.size()];
	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK_CUDA(cudaMemcpyAsync(buffers_[inputIndex_], inputCPU_, batchSize_ * inputGeometry_.area() * sizeof(float), cudaMemcpyHostToDevice, stream_));
	context_->enqueue(batchSize_, buffers_, stream_, nullptr);
	CHECK_CUDA(cudaMemcpyAsync(output, buffers_[outputIndex_], batchSize_ * labels_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
	cudaStreamSynchronize(stream_);

	return vector<float>(output, output + sizeof(output)/sizeof(output[0]));
}
#else
#include <vector>
#include "GIEClassifier.hpp"

using namespace std;
using namespace cv;

GIEClassifier::GIEClassifier(const string& modelFile,
      const string& trainedFile,
      const string& zcaWeightFile,
      const string& labelFile,
      const size_t  batchSize) :
	Classifier(modelFile, trainedFile, zcaWeightFile, labelFile, batchSize)
{
	cerr << "GIE support not available" << endl;
}

GIEClassifier::~GIEClassifier()
{
}

bool GIEClassifier::initialized(void) const
{
	return true;
}

vector<float> GIEClassifier::PredictBatch(const vector<Mat> &imgs)
{
	cerr << "GIE support not available" << endl;
	return vector<float>(imgs.size() * labels_.size(), 0.0);
}
#endif

#if 0
int main(int argc, char** argv)
{
	// create a GIE model from the caffe model and serialize it to a stream
	caffeToGIEModel("mnist.prototxt", "mnist.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, gieModelStream);

	// read a random digit file
	srand(unsigned(time(nullptr)));
	uint8_t fileData[INPUT_H*INPUT_W];
	readPGMFile(std::to_string(rand() % 10) + ".pgm", fileData);

	// print an ascii representation
	std::cout << "\n\n\n---------------------------" << "\n\n\n" << std::endl;
	for (int i = 0; i < INPUT_H*INPUT_W; i++)
		std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");

	// parse the mean file and 	subtract it from the image
	IBinaryProtoBlob* meanBlob = CaffeParser::parseBinaryProto(locateFile("mnist_mean.binaryproto").c_str());
	const float *meanData = reinterpret_cast<const float*>(meanBlob->getData());

	float data[INPUT_H*INPUT_W];
	for (int i = 0; i < INPUT_H*INPUT_W; i++)
		data[i] = float(fileData[i])-meanData[i];

	meanBlob->destroy();

	// deserialize the engine 
	gieModelStream.seekg(0, gieModelStream.beg);

	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream);

	IExecutionContext *context = engine->createExecutionContext();

	// run inference
	float prob[OUTPUT_SIZE];
	doInference(*context, data, prob, 1);

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	// print a histogram of the output distribution
	std::cout << "\n\n";
	for (unsigned int i = 0; i < 10; i++)
		std::cout << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
	std::cout << std::endl;

	return 0;
}
#endif
