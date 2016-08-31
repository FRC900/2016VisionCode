#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>

#include "Infer.h"
#include "caffeParser.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

#define CHECK_CUDA(status)								\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;

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


template <class MatT>
class CaffeClassifier 
{
private:
	IRuntime* runtime_;          // network runtime
	ICudaEngine* engine_;        // network engine
	IExecutionContext *context_; // netowrk context to run on engine
	cudaStream_t stream_;
	void* buffers_[2];           // input and output buffers
	int _inputIndex;
	int _outputIndex;
};

template <class MatT>
GIEClassifier<MatT>::GIEClassifier(const string& modelFile,
      const string& trainedFile,
      const string& zcaWeightFile,
      const string& labelFile,
      const size_t  batchSize) :
	zca_(zcaWeightFile.c_str())
{
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
	cout << "Loading GIE model " << modelFile << " " << trainedFile << " " << zcaWeightFile << " " << labelFile << endl;

	batchSize_ = batchSize;

	std::stringstream gieModelStream;
	// TODO :: read from file if exists and is newer than modelFile and trainedFile
	caffeToGIEModel(modelFile, trainedFile, std::vector < std::string > { OUTPUT_BLOB_NAME }, batchSize_, gieModelStream);

	// Create runable version of model by
	// deserializing the engine 
	gieModelStream.seekg(0, gieModelStream.beg);

	runtime_ = createInferRuntime(gLogger);
	engine_  = runtime->deserializeCudaEngine(gieModelStream);
	context_ = engine->createExecutionContext();

	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine_.getNbBindings() == 2);

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	inputIndex_  = engine.getBindingIndex(INPUT_BLOB_NAME); 
	outputIndex_ = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// create GPU buffers and a stream
	CHECK_CUDA(cudaMalloc(&buffers_[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&buffers_[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

	// create CUDA stream used to run engine
	CHECK_CUDA(cudaStreamCreate(&stream_));

	// Load labels
	// This will be used to give each index of the output
	// a human-readable name
	ifstream labels(labelFile.c_str());
	CHECK(labels) << "Unable to open labels file " << labelFile;
	string line;
	while (getline(labels, line))
		labels_.push_back(string(line));
}

GIEClassifer::~GIEClassifier()
{
	// release the stream and the buffers
	cudaStreamDestroy(stream_);
	CHECK_CUDA(cudaFree(buffers_[inputIndex_]));
	CHECK_CUDA(cudaFree(buffers_[outputIndex_]));
	context_->destroy();
	engine_->destroy();
	runtime_->destroy();
}

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
															  DataType::kFLOAT);

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

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	float output[labels_.size()];
	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK_CUDA(cudaMemcpyAsync(buffers_[inputIndex_], input, batchSize_ * inputGeometry_.area() * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers_, stream, nullptr);
	CHECK_CUDA(cudaMemcpyAsync(output, buffers_[outputIndex_], batchSize_ * labels_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
}


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
