#include "objdetect.hpp"

int scale         = 35;
int d12NmsThreshold = 40;
int d24NmsThreshold = 75;
int minDetectSize = 44;
int maxDetectSize = 450;
int d12Threshold  = 65;
int d24Threshold  = 65;

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
void GPU_NNDetect::Detect (const Mat &frameInput, const Mat &depthIn, vector<Rect> &imageRects)
{
	// Control detect threshold via sliders.
	// Hack - set D24 to 0 to bypass running it
	vector<double> detectThreshold;
	detectThreshold.push_back(d12Threshold ? d12Threshold / 100. : 0.01);
	detectThreshold.push_back(d24Threshold / 100.);

	vector<double> nmsThreshold;
	nmsThreshold.push_back(d12NmsThreshold/100.);
	nmsThreshold.push_back(d24NmsThreshold/100.);

	classifier_.detectMultiscale(frameInput,
			depthIn,
			Size(minDetectSize * DETECT_ASPECT_RATIO, minDetectSize),
			Size(maxDetectSize * DETECT_ASPECT_RATIO, maxDetectSize),
			1.01 + scale/100.,
			nmsThreshold,
			detectThreshold,
			imageRects);
}

//gpu version with wrapper to upload Mat to GpuMat
/*
void GPU_CascadeDetect::Detect (const Mat &frame, vector<Rect> &imageRects)
{
   //uploadFrame.upload(frame);
   Detect(uploadFrame, imageRects);
}*/
