#include <iostream>
#include <sys/stat.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "CaffeBatchPrediction.hpp"

using namespace std;
using namespace caffe;
using namespace cv;
using namespace gpu;

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
			while (queue_.size()==0)
				cond_.wait(lock);

			// Retrieve the data from the queue
			T result=queue_.front(); 
			queue_.pop();
			return result;
		} // Lock is automatically released here

	private:
		std::queue<T> queue_; // Use STL queue to store data
		boost::mutex mutex_; // The mutex to synchronise on
		boost::condition_variable cond_; // The condition to wait for
};
 
// Use multiple threads running neural nets.  One
// thread doesn't seem to be enough to keep the GPU
// busy.

// Each thread is constructed with the model info
// for a given net.  It then loops forever waiting
// for input data to appear on a queue. The main thread
// pushes data into the queue and the first free worker
// thread processes the data. That worker thread will
// then push results onto an output queue. After filling
// in the input data queue, the main thread loops and
// reads the results out the output queue and reassembles
// those results onto one block to return to the caller.
template <class MatT>
class CaffeClassifierThread
{
	public :
		CaffeClassifierThread(const string  &modelFile,
				const string                &trainedFile,
				const string                &meanFile,
				const int                    batchSize,
				std::shared_ptr<SynchronizedQueue<InQData<MatT>>> inQ,
				std::shared_ptr<SynchronizedQueue<OutQData>>      outQ) :
			inQ_(inQ), // input and output queues for communicating
			outQ_(outQ) // with the master thread
		{
			/* Load the network - this includes model geometry and trained weights */
			net_.reset(new Net<float>(modelFile, TEST));
			net_->CopyTrainedLayersFrom(trainedFile);

			CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
			CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

			Blob<float>* inputLayer = net_->input_blobs()[0];
			numChannels_ = inputLayer->channels();
			CHECK(numChannels_ == 3 || numChannels_ == 1)
				<< "Input layer should have 1 or 3 channels.";
			inputGeometry_ = Size(inputLayer->width(), inputLayer->height());

			/* Load the binaryproto mean file. */
			SetMean(meanFile);

			// Pre-process Mat wrapping
			inputLayer->Reshape(batchSize, numChannels_,
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

		// Get the output values for a set of images in one flat vector
		// These values will be in the same order as the labels for each
		// image, and each set of labels for an image next adjacent to the
		// one for the next image.
		// That is, [0] = value for label 0 for the first image up to 
		// [n] = value for label n for the first image. It then starts again
		// for the next image - [n+1] = label 0 for image #2.
		void operator () ()
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
				// This is the step whcih actually runs the net over
				// the input images. It produces a set of confidence scores 
				// per image - one score per label.  The higher
				// the score, the more likely the net thinks
				// it is that the input image matches a given
				// label.
				net_->ForwardPrefilled();

				// Copy the output layer to a flat vector
				Blob<float>* outputLayer = net_->output_blobs()[0];
				const float* begin = outputLayer->cpu_data();
				const float* end = begin + outputLayer->channels() * input.data_.size();
				outQ_->Enqueue(OutQData(input.batchNum_, vector<float>(begin, end)));
				// Make sure we can be interrupted
				boost::this_thread::interruption_point();
			}
		}

		Size getInputGeometry(void) const
		{
			return inputGeometry_;
		}

	private:
		/* Load the mean file in binaryproto format. */
		void SetMean(const string& meanFile) 
		{
			BlobProto blobProto;
			ReadProtoFromBinaryFileOrDie(meanFile.c_str(), &blobProto);

			/* Convert from BlobProto to Blob<float> */
			Blob<float> meanBlob;
			meanBlob.FromProto(blobProto);
			CHECK_EQ(meanBlob.channels(), numChannels_)
				<< "Number of channels of mean file doesn't match input layer.";

			/* The format of the mean file is planar 32-bit float BGR or grayscale. */
			vector<Mat> channels;
			float* data = meanBlob.mutable_cpu_data();
			for (int i = 0; i < numChannels_; ++i) 
			{
				/* Extract an individual channel. */
				Mat channel(meanBlob.height(), meanBlob.width(), CV_32FC1, data);
				channels.push_back(channel);
				data += meanBlob.height() * meanBlob.width();
			}

			/* Merge the separate channels into a single image. */
			Mat meanImg;
			merge(channels, meanImg);

			/* Compute the global mean pixel value and create a mean image
			 * filled with this value. */
			Scalar channelMean = cv::mean(meanImg);

			// Hack to possibly convert to GpuMat - if MatT is GpuMat,
			// this will upload the Mat object to a GpuMat, otherwise
			// it will just copy it to the member variable mean_
			mean_ = MatT(inputGeometry_, meanImg.type(), channelMean);
		}

		// Wrap input layer of the net into separate Mat objects
		// This sets them up to be written with actual data
		// in PreprocessBatch()
		void WrapBatchInputLayer(void)
		{
			Blob<float>* inputLayer = net_->input_blobs()[0];

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
		void SlowPreprocess(const MatT &img, MatT &output)
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
		void PreprocessBatch(const vector<MatT> &imgs)
		{
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
		// Method which returns either mutable_cpu_data or mutable_gpu_data
		// depending on whether we're using CPU or GPU Mats
		// X will always be floats for our net, but C++ syntax
		// makes us make it a generic prototype here?
		template <class X> float *GetBlobData(caffe::Blob<X> *blob);

		std::shared_ptr<caffe::Net<float> > net_; // the net itself
		Size inputGeometry_;                      // size of one input image
		int numChannels_;                         // num color channels per input image
		MatT mean_;                               // mean value of input images
		MatT sampleNormalized_;                   // working buffer for converting input images to correct format
		vector< vector<MatT> > inputBatch_;       // net input buffers wrapped in Mat's
		std::shared_ptr<SynchronizedQueue<InQData<MatT>>> inQ_; // input and output queues for communicating
		std::shared_ptr<SynchronizedQueue<OutQData>> outQ_;       // data into and out of the thread
};

// TODO : maybe don't specialize this one in case
// we use something other than Mat or GpuMat in the
// future?
template <> template <>
float *CaffeClassifierThread<Mat>::GetBlobData(Blob<float> *blob)
{
	return blob->mutable_cpu_data();
}

template <> template <>
float *CaffeClassifierThread<GpuMat>::GetBlobData(Blob<float> *blob)
{
	return blob->mutable_gpu_data();
}

static bool fileExists(const char *filename)
{
	struct stat statBuffer;
	return stat(filename, &statBuffer) == 0;
}

// Main classifier constructor.
// Checks that input files exist
// Creates multiple worker threads, each initialized with
// the net's config files
// Initializes queues to communicate with workers
template <class MatT>
CaffeClassifier<MatT>::CaffeClassifier(const string& modelFile,
      const string& trainedFile,
      const string& meanFile,
      const string& labelFile,
      int batchSize,
	  int numThreads) 
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
		Caffe::set_mode(Caffe::GPU);
	else
		Caffe::set_mode(Caffe::CPU);

	batchSize_ = batchSize;
	inQ_  = make_shared<SynchronizedQueue<InQData<MatT>>>();
	outQ_ = make_shared<SynchronizedQueue<OutQData>>();

	// Load labels.  This will be used to give each index of the output
	// a human-readable name
	ifstream labels(labelFile.c_str());
	CHECK(labels) << "Unable to open labels file " << labelFile;
	string line;
	while (getline(labels, line))
		labels_.push_back(line);

	threads_ = std::make_shared<boost::thread_group>();
	for (int i = 0; i < numThreads; i++)
	{
		CaffeClassifierThread<MatT> t(modelFile, trainedFile, meanFile, batchSize, inQ_, outQ_);
		
		inputGeometry_ = t.getInputGeometry();
		threads_->create_thread(t);
	}
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

	// Each prediction is a <label, score> pair.
	// The label is just a name of type of image
	// the net can detect and the score is the 
	// confidence the net gives that an image
	// matches that type.
	// Each image will have a vector of predictions -
	// one per label/type of image the net 
	// is trained to identify
	// Since we're dealing with mulitple images
	// per batch, there's another level of
	// vectors, one per image
	vector< vector<Prediction> > predictions;
	predictions.resize(imgs.size());

	// The code is set up to grab only the top
	// N predictions. Make sure that N doesn't
	// get larger than the number of actual
	// known labels for the net
	size_t labelsSize = labels_.size();
	numClasses = min(numClasses, labelsSize);

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
			// predictions for all images. Put it in the correct location
			// for the given batch and image
			predictions[outputBatch.batchNum_ * batchSize_ + j] = vector<Prediction>(predictionSingle);
		}
	}
	return predictions;
}

// Assorted helper functions
template <class MatT>
size_t CaffeClassifier<MatT>::BatchSize(void) const
{
	return batchSize_;
}

template <class MatT>
Size CaffeClassifier<MatT>::getInputGeometry(void) const
{
	return inputGeometry_;
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

template class CaffeClassifierThread<Mat>;
template class CaffeClassifierThread<GpuMat>;
template class CaffeClassifier<Mat>;
template class CaffeClassifier<GpuMat>;
