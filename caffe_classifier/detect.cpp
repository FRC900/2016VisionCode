#include <iostream>
#include "CaffeBatchPrediction.hpp"
#include "scalefactor.hpp"
#include "fast_nms.hpp"
#include "detect.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

static double gtod_wrapper(void)
{
   struct timeval tv;
   gettimeofday(&tv, NULL);
   return (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;
}

// TODO :: can we keep output data in GPU as well?
// Simple multi-scale detect.  Take a single image, scale it into a number
// of diffent sized images. Run a fixed-size detection window across each
// of them.  Keep track of the scale of each scaled image to map the
// detected rectangles back to the correct location and size on the
// original input images
template <class MatT>
void NNDetect<MatT>::detectMultiscale(const cv::Mat &inputImg,
	const cv::Size &minSize,
	const cv::Size &maxSize,
	std::vector<cv::Rect> &rectsOut)
{
	int wsize = classifier.getInputGeometry().width;
	std::vector<std::pair<MatT, float> > scaledimages;
	std::vector<cv::Rect> rects;
	std::vector<int> scales;
	std::vector<int> scalesOut;

	generateInitialWindows(inputImg, minSize, maxSize, wsize, scaledimages, rects, scales);
	runDetection(classifier, scaledimages, rects, scales, .7, "ball", rectsOut, scalesOut);
	for(size_t i = 0; i < rectsOut.size(); i++)
	{
		float scale = scaledimages[scalesOut[i]].second;
		rectsOut[i] = cv::Rect(rectsOut[i].x/scale, rectsOut[i].y/scale, rectsOut[i].width/scale, rectsOut[i].height/scale);
	}
}

template <class MatT>
void NNDetect<MatT>::generateInitialWindows(
      const cv::Mat  &input,
      const cv::Size &minSize,
      const cv::Size &maxSize,
      const int wsize,
      std::vector<std::pair<MatT, float> > &scaledimages,
      std::vector<cv::Rect> &rects,
      std::vector<int> &scales)
{
   rects.clear();
   scales.clear();

   // How many pixels to move the window for each step
   // TODO : figure out if it makes sense to change this depending on
   // the size of the scaled input image - i.e. it is possible that
   // a small step size on an image scaled way larger than the input
   // will end up detecting too much stuff ... each step on the larger
   // image might not correspond to a step of 1 pixel on the
   // input image?
   const int step = 6;
   //int step = std::min(img.cols, img.rows) *0.05;

   double start = gtod_wrapper(); // grab start time

   // The net expects each pixel to be 3x 32-bit floating point
   // values. Convert it once here rather than later for every
   // individual input image.
   MatT f32Img;
   MatT(input).convertTo(f32Img, CV_32FC3);

   // Create array of scaled images
   scalefactor(f32Img, cv::Size(wsize,wsize), minSize, maxSize, 1.35, scaledimages);

   // Main loop.  Look at each scaled image in turn
   for (size_t scale = 0; scale < scaledimages.size(); ++scale)
   {
	   // Start at the upper left corner.  Loop through the rows and cols until
	   // the detection window falls off the edges of the scaled image
	   for (int r = 0; (r + wsize) < scaledimages[scale].first.rows; r += step)
	   {
		   for (int c = 0; (c + wsize) < scaledimages[scale].first.cols; c += step)
		   {
			   // Save location and image data for each sub-image
			   rects.push_back(cv::Rect(c, r, wsize, wsize));
			   scales.push_back(scale);

		   }
	   }
   }
   double end = gtod_wrapper();
   std::cout << "Generate initial windows time = " << (end - start) << std::endl;
}

template <class MatT>
void NNDetect<MatT>::runDetection(CaffeClassifier<MatT> &classifier,
      const std::vector<std::pair<MatT, float> > &scaledimages,
      const std::vector<cv::Rect> &rects,
      const std::vector<int> &scales,
      float threshold,
      std::string label,
      std::vector<cv::Rect> &rectsOut,
      std::vector<int> &scalesOut)
{
   std::vector<MatT> images;
   std::vector<size_t> detected;
   int counter = 0;
   double start = gtod_wrapper(); // grab start time
   for (size_t i = 0; i < rects.size(); ++i)
   {
	   images.push_back(scaledimages[scales[i]].first(rects[i]));
	   if((images.size() == classifier.BatchSize()) || (i == rects.size() - 1))
	   {
		   doBatchPrediction(classifier, images, threshold, label, detected);
		   images.clear();
		   for(size_t j = 0; j < detected.size(); j++)
		   {
			   rectsOut.push_back(rects[counter*classifier.BatchSize() + detected[j]]);
			   scalesOut.push_back(scales[counter*classifier.BatchSize() + detected[j]]);
		   }
		   counter++;
	   }
   }
   double end = gtod_wrapper();
   std::cout << "runDetection time = " << (end - start) << std::endl;
}
// do 1 run of the classifier. This takes up batch_size predictions and adds anything found
// to the detected list
template <class MatT>
void NNDetect<MatT>::doBatchPrediction(CaffeClassifier<MatT> &classifier,
      const std::vector<MatT> &imgs,
      const float threshold,
      const std::string &label,
      std::vector<size_t> &detected)
{
   detected.clear();
   std::vector <std::vector<Prediction> >predictions = classifier.ClassifyBatch(imgs, 1);
   // Each outer loop is the predictions for one input image
   for (size_t i = 0; i < imgs.size(); ++i)
   {
	   // Each inner loop is the prediction for a particular label
	   // for the given image, sorted by score.
	   //
	   // Look for object with label <label>, > threshold confidence
	   for (std::vector <Prediction>::const_iterator it = predictions[i].begin(); it != predictions[i].end(); ++it)
	   {
		   if (it->first == label)
		   {
			   if (it->second > threshold)
			   {
				   detected.push_back(i);
			   }
			   break;
		   }
	   }
   }
}

// Explicitly instatiate classes used elsewhere
template class NNDetect<cv::Mat>;
template class NNDetect<cv::gpu::GpuMat>;
