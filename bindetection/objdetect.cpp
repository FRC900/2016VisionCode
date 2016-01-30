#include "objdetect.hpp"

int scale         = 35;
int neighbors     = 4;
int minDetectSize = 20;
int maxDetectSize = 450;

// TODO : make this a parameter to the detect code
// so that we can detect objects with different aspect ratios
const double DETECT_ASPECT_RATIO = 1.0;

using namespace std;
using namespace cv;
using namespace cv::gpu;

/*void CPU_CascadeDetect::Detect (const Mat &frame, vector<Rect> &imageRects)
{
  Mat frameGray;
  Mat frameEq;
  cvtColor( frame, frameGray, CV_BGR2GRAY );
  equalizeHist( frameGray, frameEq);

  classifier_.detectMultiScale(frameEq,
	imageRects,
	1.01 + scale/100.,
	neighbors,
	0|CV_HAAR_SCALE_IMAGE,
	Size(minDetectSize * DETECT_ASPECT_RATIO, minDetectSize),
	Size(maxDetectSize * DETECT_ASPECT_RATIO, maxDetectSize) );
}
*/
void GPU_NNDetect::Detect (const Mat &frameInput, vector<Rect> &imageRects)
{
  //cvtColor(frameGPUInput, frameGray, CV_BGR2GRAY);
  //equalizeHist(frameGray, frameEq);

  //-- Detect objects
  int detectCount;
  classifier_.detectMultiscale(frameInput,
      Size(minDetectSize * DETECT_ASPECT_RATIO, minDetectSize),
      Size(maxDetectSize * DETECT_ASPECT_RATIO, maxDetectSize),
	  1.01 + scale/100.,
      .25,
      imageRects);

  /*
  detectCount = classifier_.detectMultiScale(frameEq,
	 detectResultsGPU,
	 Size(maxDetectSize * DETECT_ASPECT_RATIO, maxDetectSize),
	 Size(minDetectSize * DETECT_ASPECT_RATIO, minDetectSize),
	 1.01 + scale/100.,
	 neighbors);
  */

  // download only detected number of rectangles
  /*
  Mat detectResult;
  detectResultsGPU.colRange(0, imageRects.size()).download(detectResult);

  imageRects.clear();
  Rect *rects = detectResult.ptr<Rect>();
  for(int i = 0; i < detectCount; ++i)
     imageRects.push_back(rects[ivoid CPU_CascadeDetect::Detect (const Mat &frame, vector<Rect> &imageRects));
  */
}

//gpu version with wrapper to upload Mat to GpuMat
/*
void GPU_CascadeDetect::Detect (const Mat &frame, vector<Rect> &imageRects)
{
   //uploadFrame.upload(frame);
   Detect(uploadFrame, imageRects);
}*/
