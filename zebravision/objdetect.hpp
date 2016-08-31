#ifndef INC_OBJDETECT_HPP__
#define INC_OBJDETECT_HPP__

#include <iostream>
#include <sys/stat.h>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "detect.hpp"

#include <vector>

// Base class for detector. Doesn't really do much - all of the heavy lifting is
// in the derived classes
class ObjDetect
{
	public :
		ObjDetect() : init_(false) {} //pass in value of false to cascadeLoadedGPU_CascadeDetect
		virtual ~ObjDetect() {}       //empty destructor
		// virtual void Detect(const cv::Mat &frame, std::vector<cv::Rect> &imageRects) = 0; //pure virtual function, must be defined by CPU and GPU detect
		virtual void Detect(const cv::Mat &frameGPUInput, const cv::Mat &depthIn, std::vector<cv::Rect> &imageRects, std::vector<cv::Rect> &uncalibImageRects)
		{
			(void)depthIn;
			(void)frameGPUInput;
			imageRects.clear();
			uncalibImageRects.clear();
		}
		bool initialized(void)
		{
			return init_;
		}

	protected:
		bool init_;
};

// CPU version of cascade classifier
/*class CPU_CascadeDetect : public ObjDetect
{
   public :
      CPU_CascadeDetect(const char *cascadeName) : ObjDetect() // call default constructor of base class
      {
	 struct stat statbuf;
	 if (stat(cascadeName, &statbuf) != 0)
	 {
	    std::cerr << "Can not open classifier input " << cascadeName << std::endl;
	    std::cerr << "Try to point to a different one with --classifierBase= ?" << std::endl;
	    return;
	 }

	 init_ = classifier_.load(cascadeName);
      }
      ~CPU_CascadeDetect(void)
      {
      }
      void Detect(const cv::Mat &frame, std::vector<cv::Rect> &imageRects); //defined elsewhere

   private :
      cv::CascadeClassifier classifier_;
};
*/
// GPU version of cascade classifier. Pretty much the same interface
// as the CPU version, but with an added method to handle data
// which is already moved to a GpuMat
class GPU_NNDetect : public ObjDetect
{
	public :
		GPU_NNDetect(Classifier *d12,
					 Classifier *d24,
					 Classifier *c12,
					 Classifier *c24,
					 float hfov) :
						ObjDetect(),
						classifier_(d12, d24, c12, c24, hfov)
		{
			/* struct stat statbuf;		
			   if (stat(cascadeName, &statbuf) != 0)
			   {
			   std::cerr << "Can not open classifier input " << cascadeName << std::endl;
			   std::cerr << "Try to point to a different one with --classifierBase= ?" << std::endl;
			   return;

			   }

			   init_ = classifier_.load(cascadeName);
			   */
			init_ = true;
		}

		~GPU_NNDetect(void)
		{
			// classifier_.release();
		}
		void Detect (const cv::Mat &frame, const cv::Mat &depthIn, std::vector<cv::Rect> &imageRects, std::vector<cv::Rect> &uncalibImageRects);
		//void Detect (const cv::gpu::GpuMat &frameGPUInput, std::vector<cv::Rect> &imageRects);

	private :
		NNDetect<cv::Mat> classifier_;
};

extern int scale;
extern int d12NmsThreshold;
extern int d24NmsThreshold;
extern int minDetectSize;
extern int maxDetectSize;
extern int d12Threshold;
extern int d24Threshold;
extern int c12Threshold;
extern int c24Threshold;

#endif
