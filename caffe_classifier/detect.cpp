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
	double scaleFactor,
	double NMSThreshold,
	std::vector<cv::Rect> &rectsOut)
{
    // Size of the first 
	int wsize = classifier.getInputGeometry().width;

    // scaled images which allow us to find images between min and max
    // size for the given classifier input window
	std::vector<std::pair<MatT, float> > scaledImages12;
	std::vector<std::pair<MatT, float> > scaledImages24;
	std::vector<std::pair<MatT, float> > scaledImages48;
    // list of initial rectangles to search. These are rectangles from
    // various scaled images above - see scales below to map to the correct
    // input image
	std::vector<cv::Rect> rects;
    // These are used to find which scaledImage a given rect comes from.
    // For rect[i], scales[i] holds an index indo scaledImages which points
    // to the correct scaled image to grab from.
	std::vector<int> scales;
	std::vector<int> scalesOut;

    // Confidence scores (0.0 - 1.0) for each detected rectangle
	std::vector<float> scores;

    // Detected is a rect, score pair. Used for non-max
    // suppression - discarding rects which overlap and 
    // keeping only the best scoring object
    std::vector<Detected> detectedOut;

    // Generate a list of initial windows to search. Each window will be 12x12 from a scaled image
    // These scaled images let us search for variable sized objects using a fixed-width detector
    MatT f32Img;
	MatT(inputImg).convertTo(f32Img, CV_32FC3);
	generateInitialWindows(f32Img, minSize, maxSize, wsize, scaleFactor, scaledImages12, rects, scales);

	// Generate scaled images for the larger detect sizes as well. Subsequent passes will use larger
    // input sizes. These images will let us grab higher res input as the detector size goes up (as
    // opposed to just scaling up the 12x12 images to a larger size).
    // 
    // scales[] maps to these scaledImages arrays in addition to the original scaledImages12, so no
    //    need to regen that for each 
	scalefactor(f32Img, cv::Size(wsize*2,wsize*2), minSize, maxSize, scaleFactor, scaledImages24);
	scalefactor(f32Img, cv::Size(wsize*4,wsize*4), minSize, maxSize, scaleFactor, scaledImages48);

	size_t i = 545;
	cv::Rect rect12 = rects[i];
	double scale12 = scaledImages12[scales[i]].second;
	std::cout << rect12 <<  " scale12 = " << scale12 << std::endl;
	cv::Rect rectOut = cv::Rect(rect12.x/scale12, rect12.y/scale12, rect12.width/scale12, rect12.height/scale12);
std::cout << rectOut << "On original image" << std::endl;
    
	double scale24 = scaledImages24[scales[i]].second;
	cv::Rect rect24(rect12.x * 2, rect12.y * 2, rect12.width * 2, rect12.height * 2);
	std::cout << rect24 <<  " scale24 = " << scale24 << std::endl;
	rectOut = cv::Rect(rect24.x/scale24, rect24.y/scale24, rect24.width/scale24, rect24.height/scale24);
	std::cout << rectOut << "On original image" << std::endl;
#if 0
	runDetection(classifier, scaledImages12, rects, scales, .4, "ball", rectsOut, scalesOut, scores);
	for(size_t i = 0; i < rectsOut.size(); i++)
	{
		float scale = scaledImages12[scalesOut[i]].second;
		rectsOut[i] = cv::Rect(rectsOut[i].x/scale, rectsOut[i].y/scale, rectsOut[i].width/scale, rectsOut[i].height/scale);
		detectedOut.push_back(Detected(rectsOut[i], scores[i]));
	}
	if (NMSThreshold > 0)
			fastNMS(detectedOut, NMSThreshold, rectsOut);
#endif
}

template <class MatT>
void NNDetect<MatT>::generateInitialWindows(
      const MatT &input,
      const cv::Size &minSize,
      const cv::Size &maxSize,
      int wsize,
      double scaleFactor,
      std::vector<std::pair<MatT, float> > &scaledImages,
      std::vector<cv::Rect> &rects,
      std::vector<int> &scales)
{
   rects.clear();
   scales.clear();

   // How many pixels to move the window for each step
   // We use 4 - the calibration step can adjust +/- 2 pixels
   // in each direction, which means they will correct for
   // anything which is actually centered in one of the
   // pixels we step over.
   const int step = 4;

   double start = gtod_wrapper(); // grab start time

   // Create array of scaled images
   scalefactor(input, cv::Size(wsize,wsize), minSize, maxSize, scaleFactor, scaledImages);

   // Main loop.  Look at each scaled image in turn
   for (size_t scale = 0; scale < scaledImages.size(); ++scale)
   {
	   // Start at the upper left corner.  Loop through the rows and cols until
	   // the detection window falls off the edges of the scaled image
	   for (int r = 0; (r + wsize) < scaledImages[scale].first.rows; r += step)
	   {
		   for (int c = 0; (c + wsize) < scaledImages[scale].first.cols; c += step)
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
      std::vector<int> &scalesOut,
      std::vector<float> &scores)
{
   // Accumulate a number of images to test and pass them in to
   // the NN prediction as a batch
   std::vector<MatT> images;

   // Return value from detection. This is a list of indexes from
   // the input which have a high enough confidence score
   std::vector<size_t> detected;

   int counter = 0;
   double start = gtod_wrapper(); // grab start time

   for (size_t i = 0; i < rects.size(); ++i)
   {
	   images.push_back(scaledimages[scales[i]].first(rects[i]));
	   if((images.size() == classifier.BatchSize()) || (i == rects.size() - 1))
	   {
		   doBatchPrediction(classifier, images, threshold, label, detected, scores);
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
      float threshold,
      const std::string &label,
      std::vector<size_t> &detected,
      std::vector<float>  &scores)
{
   detected.clear();
   std::vector <std::vector<Prediction> >predictions = classifier.ClassifyBatch(imgs, 2);
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
				   scores.push_back(it->second);
			   }
			   break;
		   }
	   }
   }
}

// Explicitly instatiate classes used elsewhere
template class NNDetect<cv::Mat>;
template class NNDetect<cv::gpu::GpuMat>;
