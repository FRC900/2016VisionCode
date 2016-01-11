#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "CaffeBatchPrediction.hpp"

template <class MatT>
CaffeClassifier<MatT>::CaffeClassifier(const std::string& model_file,
      const std::string& trained_file,
      const std::string& mean_file,
      const std::string& label_file,
      const int batch_size) {

   if (IsGPU())
   	caffe::Caffe::set_mode(caffe::Caffe::GPU);
   else
   	caffe::Caffe::set_mode(caffe::Caffe::CPU);

   /* Set batchsize */
   batch_size_ = batch_size;

   /* Load the network - this includes model geometry and trained weights */
   net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
   net_->CopyTrainedLayersFrom(trained_file);

   CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
   CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

   caffe::Blob<float>* input_layer = net_->input_blobs()[0];
   num_channels_ = input_layer->channels();
   CHECK(num_channels_ == 3 || num_channels_ == 1)
      << "Input layer should have 1 or 3 channels.";
   input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

   /* Load the binaryproto mean file. */
   SetMean(mean_file);

   /* Load labels. */
   // This will be used to give each index of the output
   // a human-readable name
   std::ifstream labels(label_file.c_str());
   CHECK(labels) << "Unable to open labels file " << label_file;
   std::string line;
   while (std::getline(labels, line))
      labels_.push_back(std::string(line));

   caffe::Blob<float>* output_layer = net_->output_blobs()[0];
   CHECK_EQ(labels_.size(), output_layer->channels())
      << "Number of labels is different from the output layer dimension.";

   // Pre-process Mat wrapping
   input_layer->Reshape(batch_size_, num_channels_,
	 input_geometry_.height,
	 input_geometry_.width);

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
static bool PairCompare(const std::pair<float, int>& lhs, 
			const std::pair<float, int>& rhs) 
{
   return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) 
{
   std::vector<std::pair<float, int> > pairs;
   for (size_t i = 0; i < v.size(); ++i)
      pairs.push_back(std::make_pair(v[i], i));
   std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

   std::vector<int> result;
   for (size_t i = 0; i < N; ++i)
      result.push_back(pairs[i].second);
   return result;
}

// Given X input images, return X vectors of predictions.
// Each of the X vectors are themselves a vector which will have the 
// N predictions with the highest confidences for the corresponding
// input image
template <class MatT>
std::vector< std::vector<Prediction> > CaffeClassifier<MatT>::ClassifyBatch(const std::vector< MatT > &imgs, 
                                                                      size_t num_classes)
{
   // output_batch will be a flat vector of N floating point values 
   // per image (1 per N output labels), repeated
   // times the number of input images batched per run
   // Convert that into the output vector of vectors
   std::vector<float> output_batch = PredictBatch(imgs);
   std::vector< std::vector<Prediction> > predictions;
   size_t labels_size = labels_.size();
   num_classes = std::min(num_classes, labels_size);

   // For each image, find the top num_classes values
   for(size_t j = 0; j < imgs.size(); j++)
   {
      // Create an output vector just for values for this image. Since
      // each image has labels_size values, that's output_batch[j*labels_size]
      // through output_batch[(j+1) * labels_size]
      std::vector<float> output(output_batch.begin() + j*labels_size, output_batch.begin() + (j+1)*labels_size);
      // For the output specific to the jth image, grab the
      // indexes of the top num_classes predictions
      std::vector<int> maxN = Argmax(output, num_classes);
      // Using those top N indexes, create a set of labels/value predictions
      // specific to this jth image
      std::vector<Prediction> prediction_single;
      for (size_t i = 0; i < num_classes; ++i) 
      {
	 int idx = maxN[i];
	 prediction_single.push_back(std::make_pair(labels_[idx], output[idx]));
      }
      // Add the predictions for this image to the list of
      // predictions for all images
      predictions.push_back(std::vector<Prediction>(prediction_single));
   }
   return predictions;
}

/* Load the mean file in binaryproto format. */
template <class MatT>
void CaffeClassifier<MatT>::SetMean(const std::string& mean_file) 
{
   caffe::BlobProto blob_proto;
   ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

   /* Convert from BlobProto to Blob<float> */
   caffe::Blob<float> mean_blob;
   mean_blob.FromProto(blob_proto);
   CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

   /* The format of the mean file is planar 32-bit float BGR or grayscale. */
   std::vector<cv::Mat> channels;
   float* data = mean_blob.mutable_cpu_data();
   for (int i = 0; i < num_channels_; ++i) 
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
   mean_ = MatT(input_geometry_, mean.type(), channel_mean);
}

// TODO : see if we can do this once at startup or if
// it has to be done each pass.  If it can be done once,
// we can wrap the nets in Mat arrays in the constructor 
// and re-use them multiple times?
template <class MatT>
void CaffeClassifier<MatT>::setBatchSize(size_t batch_size) 
{
   CHECK(batch_size >= 0);
   if (batch_size == batch_size_) return;
   batch_size_ = batch_size;
   reshapeNet();
}

template <class MatT>
void CaffeClassifier<MatT>::reshapeNet() 
{
   CHECK(net_->input_blobs().size() == 1);
   caffe::Blob<float>* input_layer = net_->input_blobs()[0];
   input_layer->Reshape(batch_size_, num_channels_,
	 input_geometry_.height,
	 input_geometry_.width);
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
std::vector<float> CaffeClassifier<MatT>::PredictBatch(const std::vector<MatT> &imgs) 
{
   // Process each image so they match the format
   // expected by the net, then copy the images
   // into the net's input buffers
   PreprocessBatch(imgs);

   // Run a forward pass with the data filled in from above
   net_->ForwardPrefilled();

   /* Copy the output layer to a flat std::vector */
   caffe::Blob<float>* output_layer = net_->output_blobs()[0];
   const float* begin = output_layer->cpu_data();
   const float* end = begin + output_layer->channels()*imgs.size();
   return std::vector<float>(begin, end);
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
   input_batch.clear();
   for ( int j = 0; j < num; j++)
   {
      std::vector<MatT> input_channels;
      for (int i = 0; i < input_layer->channels(); ++i)
      {
	 MatT channel(height, width, CV_32FC1, input_data);
	 input_channels.push_back(channel);
	 input_data += width * height;
      }
      input_batch.push_back(std::vector<MatT>(input_channels));
   }
}

// Slow path for Preprocess - used if the image has to be
// color converted, resized, or resampled to float
template <class MatT>
void CaffeClassifier<MatT>::SlowPreprocess(const MatT &img, MatT &output)
{
   std::cout<< "In slow path " << std::endl;
      /* Convert the input image to the input image format of the network. */
      MatT sample;
      if (img.channels() == 3 && num_channels_ == 1)
	 cvtColor(img, sample, CV_BGR2GRAY);
      else if (img.channels() == 4 && num_channels_ == 1)
	 cvtColor(img, sample, CV_BGRA2GRAY);
      else if (img.channels() == 4 && num_channels_ == 3)
	 cvtColor(img, sample, CV_BGRA2BGR);
      else if (img.channels() == 1 && num_channels_ == 3)
	 cvtColor(img, sample, CV_GRAY2BGR);
      else
	 sample = img;

      MatT sample_resized;
      if (sample.size() != input_geometry_)
	 resize(sample, sample_resized, input_geometry_);
      else
	 sample_resized = sample;

      MatT sample_float;
      if (num_channels_ == 3)
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
void CaffeClassifier<MatT>::PreprocessBatch(const std::vector<MatT> &imgs)
{
   for (int i = 0 ; i < imgs.size(); i++)
   {
      if ((imgs[i].channels() != num_channels_) ||
	  (imgs[i].size()     != input_geometry_) ||
	  (num_channels_ == 3) && (imgs[i].type() != CV_32FC3) ||
	  (num_channels_ == 1) && (imgs[i].type() != CV_32FC1))
	 SlowPreprocess(imgs[i], sample_normalized_);
      else
	 subtract(imgs[i], mean_, sample_normalized_);

      /* This operation will write the separate BGR planes directly to the
       * input layer of the network because it is wrapped by the MatT
       * objects in input_channels. */
      std::vector<MatT> *input_channels = &input_batch.at(i);
      split(sample_normalized_, *input_channels);

#if 1
      // TODO : CPU Mats + GPU Caffe fails if this isn't here, no idea why
      if (i ==0 )
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
   return batch_size_;
}

template <class MatT>
cv::Size CaffeClassifier<MatT>::getInputGeometry(void) const
{
   return input_geometry_;
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
   return false;
}

template <>
bool CaffeClassifier<cv::gpu::GpuMat>::IsGPU(void) const
{
   return true;
}

template class CaffeClassifier<cv::Mat>;
template class CaffeClassifier<cv::gpu::GpuMat>;
