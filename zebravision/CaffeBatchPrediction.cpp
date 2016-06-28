#include <iostream>
#include <sys/stat.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "CaffeBatchPrediction.hpp"

using namespace std;

#if 0
// Queue class that has thread synchronisation
template <typename T>
class SynchronisedQueue
{
	public:
		// Add data to the queue and notify others
		void Enqueue(const T& data)
		{
			// Acquire lock on the queue
			boost::unique_lock<boost::mutex> lock(m_mutex);

			// Add the data to the queue
			m_queue.push(data);

			// Notify others that data is ready
			m_cond.notify_one();
		} // Lock is automatically released here

		// Get data from the queue. Wait for data if not available
		T Dequeue(void)
		{
			// Acquire lock on the queue
			boost::unique_lock<boost::mutex> lock(m_mutex);

			// When there is no data, wait till someone fills it.
			// Lock is automatically released in the wait and obtained
			// again after the wait
			while (m_queue.size()==0)
				m_cond.wait(lock);

			// Retrieve the data from the queue
			T result=m_queue.front(); 
			m_queue.pop();
			return result;
		} // Lock is automatically released here

	private:
		queue<T> m_queue; // Use STL queue to store data
		boost::mutex m_mutex; // The mutex to synchronise on
		boost::condition_variable m_cond; // The condition to wait for
};
 

template <class MatT>
class CaffeClassifierThread
{
	public :
		CaffeClassiferThread(const string& modelFile,
				const string& trainedFile,
				const string& meanFile,
				const string& labelFile,
				const int batchSize,
				SynchronizedQueue<pair<int,vector<MatT>> *inQ,
				SynchronizedQueue<pair<int,vector<float>> *outQ)
		{}

		// The thread function reads data from the queue
		void operator () ()
		{
			while (true)
			{
				// Get the data from the queue, blocking if none
				// is available
				pair<int, vector<MatT>> input = inQ_->Dequeue();
				// Process each image so they match the format
				// expected by the net, then copy the images
				// into the net's input buffers
				PreprocessBatch(input.second);
				// Run a forward pass with the data filled in from above
				net_->ForwardPrefilled();
				/* Copy the output layer to a flat vector */
				caffe::Blob<float>* output_layer = net_->output_blobs()[0];
				const float* begin = output_layer->cpu_data();
				const float* end = begin + output_layer->channels()*imgs.size();
				outQ->Enqueue(make_pair<intput.first, vector<float>(begin, end) >);
				// Make sure we can be interrupted
				boost::this_thread::interruption_point();
			}
		}
};

CaffeClassifier constructor creates a thread pool of these threads

// Given X input images, return X vectors of predictions.
// Each of the X vectors are themselves a vector which will have the 
// N predictions with the highest confidences for the corresponding
// input image
template <class MatT>
vector< vector<Prediction> > CaffeClassifier<MatT>::ClassifyBatch(
		const vector< MatT > &imgs, 
		size_t num_classes)
{
	int batchCount = 0;
    pair<batch,vector<MatT>> imgBatch;
    for (auto it = imgs.cbegin(); it != imgs.cend(); ++it)
    {
		imgBatch.second.push_back(*it);
		if ((imgBatch.size() == batchSize_) || ((it + 1) == imgs.cend()))
		{
			// Enqueue this image batch. The call will kick
			// off a worker thread if one is idle
			imgBatch.first = batchCount++;
			imgQ.Enqueue(imgBatch);
			imgBatch.first.clear();
		}
    }

	vector< vector<Prediction> > predictions;
	predictions.size(imgs.size());
	size_t labels_size = labels_.size();
	num_classes = min(num_classes, labels_size);

	while (--batchCount)
	{
	// output_batch will be a flat vector of N floating point values 
	// per image (1 per N output labels), repeated
	// times the number of input images batched per run
	// Convert that into the output vector of vectors
	pair<int, vector<float>> output_batch = outQ.Dequeue();

	// For each image, find the top num_classes values
	for(size_t j = 0; j < imgs.size(); j++)
	{
		// Create an output vector just for values for this image. Since
		// each image has labels_size values, that's output_batch[j*labels_size]
		// through output_batch[(j+1) * labels_size]
		vector<float> output(output_batch.first.begin() + j*labels_size, output_batch.first.begin() + (j+1)*labels_size);
		// For the output specific to the jth image, grab the
		// indexes of the top num_classes predictions
		vector<int> maxN = Argmax(output, num_classes);
		// Using those top N indexes, create a set of labels/value predictions
		// specific to this jth image
		vector<Prediction> prediction_single;
		for (size_t i = 0; i < num_classes; ++i) 
		{
			int idx = maxN[i];
			prediction_single.push_back(make_pair(labels_[idx], output[idx]));
		}
		// Add the predictions for this image to the list of
		// predictions for all images. Put it in the correct location
		// for the given batch and image
		predictions[output_batch.first + batchSize_ + j] = vector<Prediction>(prediction_single);
	}
	}
	return predictions;
}

#endif
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

	caffe::Blob<float>* input_layer = net_->input_blobs()[0];
	numChannels_ = input_layer->channels();
	CHECK(numChannels_ == 3 || numChannels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	inputGeometry_ = cv::Size(input_layer->width(), input_layer->height());

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

	caffe::Blob<float>* output_layer = net_->output_blobs()[0];
	CHECK_EQ(labels_.size(), output_layer->channels())
		<< "Number of labels is different from the output layer dimension.";

	// Pre-process Mat wrapping
	input_layer->Reshape(batchSize_, numChannels_,
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
		size_t num_classes)
{
	// output_batch will be a flat vector of N floating point values 
	// per image (1 per N output labels), repeated
	// times the number of input images batched per run
	// Convert that into the output vector of vectors
	vector<float> output_batch = PredictBatch(imgs);
	vector< vector<Prediction> > predictions;
	size_t labels_size = labels_.size();
	num_classes = min(num_classes, labels_size);
	// For each image, find the top num_classes values
	for(size_t j = 0; j < imgs.size(); j++)
	{
		// Create an output vector just for values for this image. Since
		// each image has labels_size values, that's output_batch[j*labels_size]
		// through output_batch[(j+1) * labels_size]
		vector<float> output(output_batch.begin() + j*labels_size, output_batch.begin() + (j+1)*labels_size);
		// For the output specific to the jth image, grab the
		// indexes of the top num_classes predictions
		vector<int> maxN = Argmax(output, num_classes);
		// Using those top N indexes, create a set of labels/value predictions
		// specific to this jth image
		vector<Prediction> prediction_single;
		for (size_t i = 0; i < num_classes; ++i) 
		{
			int idx = maxN[i];
			prediction_single.push_back(make_pair(labels_[idx], output[idx]));
		}
		// Add the predictions for this image to the list of
		// predictions for all images
		predictions.push_back(vector<Prediction>(prediction_single));
	}
	return predictions;
}

/* Load the mean file in binaryproto format. */
template <class MatT>
void CaffeClassifier<MatT>::SetMean(const string& meanFile) 
{
	caffe::BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(meanFile.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	caffe::Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), numChannels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < numChannels_; ++i) 
	{
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	 * filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);

	// Hack to possibly convert to GpuMat - if MatT is GpuMat,
	// this will upload the Mat object to a GpuMat, otherwise
	// it will just copy it to the member variable mean_
	mean_ = MatT(inputGeometry_, mean.type(), channel_mean);
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
	caffe::Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(batchSize_, numChannels_,
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
	caffe::Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels()*imgs.size();
	return vector<float>(begin, end);
}

// Wrap input layer of the net into separate Mat objects
// This sets them up to be written with actual data
// in PreprocessBatch()
// TODO : handle 
template <class MatT>
void CaffeClassifier<MatT>::WrapBatchInputLayer(void)
{
	caffe::Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	int num = input_layer->num();
	float* input_data = GetBlobData(input_layer);
	inputBatch_.clear();
	for ( int j = 0; j < num; j++)
	{
		vector<MatT> input_channels;
		for (int i = 0; i < input_layer->channels(); ++i)
		{
			MatT channel(height, width, CV_32FC1, input_data);
			input_channels.push_back(channel);
			input_data += width * height;
		}
		inputBatch_.push_back(vector<MatT>(input_channels));
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

	MatT sample_resized;
	if (sample.size() != inputGeometry_)
		resize(sample, sample_resized, inputGeometry_);
	else
		sample_resized = sample;

	MatT sample_float;
	if (((numChannels_ == 3) && (sample_resized.type() == CV_32FC3)) ||
		((numChannels_ == 1) && (sample_resized.type() == CV_32FC1)) )
		sample_float = sample_resized;
	else if (numChannels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	subtract(sample_float, mean_, output);
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
		 * objects in input_channels. */
		vector<MatT> *input_channels = &inputBatch_.at(i);
		split(sampleNormalized_, *input_channels);

#if 1
		// TODO : CPU Mats + GPU Caffe fails if this isn't here, no idea why
		if (i == 0)
			CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
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
