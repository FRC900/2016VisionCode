#include <iostream>
#include <sstream>
#include "CaffeBatchPrediction.hpp"
#include "scalefactor.hpp"
#include "fast_nms.hpp"
#include "detect.hpp"
#include "zedin.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

static double gtod_wrapper(void)
{
   struct timeval tv;
   gettimeofday(&tv, NULL);
   return (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;
}

int main(int argc, char *argv[])
{
   if (argc != 10) 
   {
      std::cerr << "Usage: " << argv[0]
	 << " d12-deploy.prototxt d12-network.caffemodel"
	 << " d12-mean.binaryproto d12-labels.txt " 
	 << " d24-deploy.prototxt d24-network.caffemodel"
	 << " d24-mean.binaryproto d24-labels.txt img.jpg" << std::endl;
      return 1;
   }
   ::google::InitGoogleLogging(argv[0]);
   std::vector<std::string> d12Info;
   std::vector<std::string> d24Info;
   d12Info.push_back(argv[1]);
   d12Info.push_back(argv[2]);
   d12Info.push_back(argv[3]);
   d12Info.push_back(argv[4]);
   d24Info.push_back(argv[5]);
   d24Info.push_back(argv[6]);
   d24Info.push_back(argv[7]);
   d24Info.push_back(argv[8]);
   std::string file = argv[9];
   Mat frame;
   ZedIn* cap;
   cap = new ZedIn(argv[5]);
   if(!cap->getNextFrame(frame, false))
   {
      std::cerr << "err" << std::endl;
      return 1;
   }
   NNDetect<cv::Mat> detect(d12Info, d24Info);
   cv::Mat emptyMat;
   cv::Size minSize(40,40);
   cv::Size maxSize(700,700);
   std::vector<cv::Rect> rectsOut;
   std::vector<cv::Rect> depthRectsOut;
   std::vector<double> detectThresholds;
   detectThresholds.push_back(0.75);
   detectThresholds.push_back(0.5);
   Mat depthMat;
   while(1)
   {
   cap->getNextFrame(frame, false);
   cap->getDepthMat(depthMat);
   if(frame.empty())
   {
	break;
   }
   // min and max size of object we're looking for.  The input
   // image will be scaled so that these min and max sizes
   // line up with the classifier input size.  Other scales will
   // fill in the range between those two end points.
   detect.detectMultiscale(frame, emptyMat, minSize, maxSize, 1.15, 0.4, detectThresholds, rectsOut);
   detect.detectMultiscale(frame, depthMat, minSize, maxSize, 1.15, 0.4, detectThresholds, rectsOut);
   namedWindow("Image", cv::WINDOW_AUTOSIZE);
   for (std::vector<cv::Rect>::const_iterator it = rectsOut.begin(); it != rectsOut.end(); ++it)
      rectangle(frame, *it, cv::Scalar(0,0,255));
   for (std::vector<cv::Rect>::const_iterator it = depthRectsOut.begin(); it != depthRectsOut.end(); ++it)
      rectangle(frame, *it, cv::Scalar(255,0,0));
   imshow("Image", frame);
   //imwrite("detect.png", inputImg);
   cv::waitKey(0);
   return 0;
   }
}

