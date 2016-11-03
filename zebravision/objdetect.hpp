#pragma once

#include "opencv2_3_shim.hpp"
#include "detect.hpp"
#if CV_MAJOR_VERSION == 2
#define cuda gpu
#else
#include <opencv2/cudaobjdetect.hpp>
#endif
#ifndef GIE
#include "CaffeClassifier.hpp"
#else
#include "GIEClassifier.hpp"
#endif

#include <vector>

// Base class for detector. Doesn't really do much - all 
// of the heavy lifting is in the derived classes
class ObjDetect
{
	public :
		ObjDetect(void);
		virtual ~ObjDetect();
		// Call to detect objects.  Takes frameInput as RGB
		// image and optional depthIn which holds matching
		// depth data for each RGB pixel.
		// Returns a set of detected rectangles.
		virtual void Detect(const cv::Mat &frameInput, 
							const cv::Mat &depthIn, 
							std::vector<cv::Rect> &imageRects, 
							std::vector<cv::Rect> &uncalibImageRects) = 0;
		virtual std::vector<size_t> DebugInfo(void) const;
		bool initialized(void) const;

	protected:
		bool init_;
};

// Code for calling LBP/HAAR cascade classifier
// Mainly for backwards compatibility with 2015 vision code
class ObjDetectCascadeCPU: public ObjDetect
{
	public : 
		ObjDetectCascadeCPU(const std::string &cascadeName);
		void Detect(const cv::Mat &frameIn, 
					const cv::Mat &depthIn, 
					std::vector<cv::Rect> &imageRects, 
					std::vector<cv::Rect> &uncalibImageRects);
	private:
		cv::CascadeClassifier classifier_;
};


// Class for GPU version of cascade classifier
class ObjDetectCascadeGPU : public ObjDetect
{
	public : 
		ObjDetectCascadeGPU(const std::string &cascadeName);
		void Detect(const cv::Mat &frameIn, 
					const cv::Mat &depthIn, 
					std::vector<cv::Rect> &imageRects, 
					std::vector<cv::Rect> &uncalibImageRects);
	private:
		cv::Ptr<cv::cuda::CascadeClassifier> classifier_;
};

// Class to handle detections for all NNet based
// detectors. Detect code is the same for all of
// them even though the detector and classifier
// types are different.  Code the common Detect call
// here and then create the various classifiers in a
// set of derived classes
template <class MatT, class ClassifierT>
class ObjDetectNNet : public ObjDetect
{
	public:
		ObjDetectNNet(std::vector<std::string> &d12Files,
					  std::vector<std::string> &d24Files,
					  std::vector<std::string> &c12Files,
					  std::vector<std::string> &c24Files,
					  float hfov);
		void Detect(const cv::Mat &frameIn, 
					const cv::Mat &depthIn, 
					std::vector<cv::Rect> &imageRects, 
					std::vector<cv::Rect> &uncalibImageRects);
		std::vector<size_t> DebugInfo(void) const;
	private :
		NNDetect<MatT, ClassifierT> classifier_;
};

#ifndef GIE
// All-CPU code
class ObjDetectCaffeCPU : public ObjDetectNNet<cv::Mat, CaffeClassifier<cv::Mat>>
{
	public :
		ObjDetectCaffeCPU(std::vector<std::string> &d12Files,
							 std::vector<std::string> &d24Files,
							 std::vector<std::string> &c12Files,
							 std::vector<std::string> &c24Files,
							 float hfov) :
						ObjDetectNNet(d12Files, d24Files, c12Files, c24Files, hfov)
		{ }
};

// Both detector and Caffe run on the GPU
class ObjDetectCaffeGPU : public ObjDetectNNet<GpuMat, CaffeClassifier<GpuMat>>
{
	public :
		ObjDetectCaffeGPU(std::vector<std::string> &d12Files,
							 std::vector<std::string> &d24Files,
							 std::vector<std::string> &c12Files,
							 std::vector<std::string> &c24Files,
							 float hfov) :
						ObjDetectNNet(d12Files, d24Files, c12Files, c24Files, hfov)
		{ }
};
#else
// Detector does resizing, sliding windows and so on
// in CPU.  GIE run on GPU
class ObjDetectTensorRTCPU : public ObjDetectNNet<cv::Mat, GIEClassifier<cv::Mat>>
{
	public :
		ObjDetectTensorRTCPU(std::vector<std::string> &d12Files,
							 std::vector<std::string> &d24Files,
							 std::vector<std::string> &c12Files,
							 std::vector<std::string> &c24Files,
							 float hfov) :
						ObjDetectNNet(d12Files, d24Files, c12Files, c24Files, hfov)
		{ }
};

// Both detector and GIE run on the GPU
class ObjDetectTensorRTGPU : public ObjDetectNNet<GpuMat, GIEClassifier<GpuMat>>
{
	public :
		ObjDetectTensorRTGPU(std::vector<std::string> &d12Files,
							 std::vector<std::string> &d24Files,
							 std::vector<std::string> &c12Files,
							 std::vector<std::string> &c24Files,
							 float hfov) :
						ObjDetectNNet(d12Files, d24Files, c12Files, c24Files, hfov)
		{ }
};
#endif

// Various globals controlling detection.  
extern int scale;
extern int d12NmsThreshold;
extern int d24NmsThreshold;
extern int minDetectSize;
extern int maxDetectSize;
extern int d12Threshold;
extern int d24Threshold;
extern int c12Threshold;
extern int c24Threshold;
