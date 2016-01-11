#ifndef INC__CAFFEBATCHPREDICTION_HPP_
#define INC__CAFFEBATCHPREDICTION_HPP_

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<std::string, float> Prediction;

// CPU and GPU code is basically the same, so make the matrix
// type used a template parameter.
// For a CPU classifier, use CaffeClassifer<cv::Mat> fooCPU, and
// use CaffeClassifier<cv::gpu::GpuMat> fooGPU
template <class MatT>
class CaffeClassifier {
 public:
  CaffeClassifier(const std::string& model_file,
             const std::string& trained_file,
             const std::string& mean_file,
             const std::string& label_file,
             const int batch_size);

  // Given X input images, return X vectors of predictions.
  // Each prediction is a label, value pair, where the value is
  // the confidence level for each given label.
  // Each of the X vectors are themselves a vector which will have the 
  // N predictions with the highest confidences for the corresponding
  // input image
  std::vector< std::vector<Prediction> > ClassifyBatch(const std::vector< MatT > &imgs, size_t num_classes);

  // Get the width and height of an input image to the net
  cv::Size getInputGeometry(void) const;

  // Get the batch size of the model
  size_t BatchSize(void) const;

  // Change the batch size of the model on the fly
  void setBatchSize(size_t batch_size);

  // Get an image with the mean value of all of the training images
  const MatT getMean(void) const;

 private:
  // Load the mean image from a file, set mean_ member var
  // with it.
  void SetMean(const std::string& mean_file);

  // Helper to resize the net
  void reshapeNet(void);

  // Get the output values for a set of images
  // These values will be in the same order as the labels for each
  // image, and each set of labels for an image next adjacent to the
  // one for the next image.
  // That is, [0] = value for label 0 for the first image up to 
  // [n] = value for label n for the first image. It then starts again
  // for the next image - [n+1] = label 0 for image #2.
  std::vector<float> PredictBatch(const std::vector< MatT > &imgs);

  // Wrap input layer of the net into separate Mat objects
  // This sets them up to be written with actual data
  // in PreprocessBatch()
  void WrapBatchInputLayer(void);

  // Take each image in Mat, convert it to the correct image type,
  // color depth, size to match the net input. Convert to 
  // F32 type, since that's what the net inputs are. 
  // Subtract out the mean before passing to the net input
  // Then actually write the images to the net input memory buffers
  void PreprocessBatch(const std::vector< MatT > &imgs);
  void SlowPreprocess(const MatT &img, MatT &output);

  // Method which returns either mutable_cpu_data or mutable_gpu_data
  // depending on whether we're using CPU or GPU Mats
  // X will always be floats for our net, but C++ syntax
  // makes us make it a generic prototype here?
  template <class X>
  float *GetBlobData(caffe::Blob<X> *blob);

  // Method specialized to return either true or false depending
  // on whether we're using GpuMats or Mats
  bool IsGPU(void) const;

  std::shared_ptr<caffe::Net<float> > net_; // the net itself
  cv::Size input_geometry_;         // size of one input image
  int num_channels_;                // num color channels per input image
  size_t batch_size_;               // number of images to process in one go
  MatT mean_;                       // mean value of input images
  MatT sample_normalized_;          // 
  std::vector<std::string> labels_; // labels for each output value
  std::vector< std::vector<MatT> > input_batch; // net input buffers wrapped in Mat's
};

#endif
