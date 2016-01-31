#include "objdetect.hpp"

int scale         = 35;
int nmsThreshold  = 39;
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
  classifier_.detectMultiscale(frameInput,
      Size(minDetectSize * DETECT_ASPECT_RATIO, minDetectSize),
      Size(maxDetectSize * DETECT_ASPECT_RATIO, maxDetectSize),
	  1.01 + scale/100.,
      .01 + nmsThreshold/100.,
      imageRects);
}

//gpu version with wrapper to upload Mat to GpuMat
/*
void GPU_CascadeDetect::Detect (const Mat &frame, vector<Rect> &imageRects)
{
   //uploadFrame.upload(frame);
   Detect(uploadFrame, imageRects);
}*/
