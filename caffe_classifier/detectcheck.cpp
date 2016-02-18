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
   if (argc != 6) 
   {
      std::cerr << "Usage: " << argv[0]
	 << " deploy.prototxt network.caffemodel"
	 << " mean.binaryproto labels.txt vid.svo" << std::endl;
      return 1;
   }
   ::google::InitGoogleLogging(argv[0]);
   std::string model_file   = argv[1];
   std::string trained_file = argv[2];
   std::string mean_file    = argv[3];
   std::string label_file   = argv[4];
   std::string file         = argv[5];
   Mat frame;
   ZedIn* cap;
   cap = new ZedIn(argv[5]);
   if(!cap->getNextFrame(frame, false))
   {
      std::cerr << "err" << std::endl;
      return 1;
   }
   NNDetect<cv::Mat> detect(model_file, trained_file, mean_file, label_file);
   cv::Mat emptyMat;
   cv::Size minSize(30,30);
   cv::Size maxSize(700,700);
   std::vector<cv::Rect> rectsOut;
   std::vector<cv::Rect> depthRectsOut;
   std::vector<double> detectThresholds;
   detectThresholds.push_back(0.75);
   detectThresholds.push_back(0.75);
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
   detect.detectMultiscale(frame, depthMat, minSize, maxSize, 1.15, 0.4, detectThresholds, depthRectsOut);
   namedWindow("Image", cv::WINDOW_AUTOSIZE);
   for (std::vector<cv::Rect>::const_iterator it = rectsOut.begin(); it != rectsOut.end(); ++it)
   {
      std::cout << "TL: " << it->tl() << std::endl;
      std::cout << "Depth: " << cap->getDepth((it->tl().x+it->br().x)/2, (it->tl().y+it->br().y)/2) << std::endl;
      std::cout << "Allowable Mid: " << (192.9 * pow(((float)it->width*(float)it->height)/((float)cap->width()*(float)cap->height()), -.534)) << std::endl;
      rectangle(frame, *it, cv::Scalar(0,0,255));
   }
   for (std::vector<cv::Rect>::const_iterator it = depthRectsOut.begin(); it != depthRectsOut.end(); ++it)
   {
      std::cout << "Made it through!" << std::endl;
      rectangle(frame, *it, cv::Scalar(255,0,0));
   }
   imshow("Image", frame);
   //imwrite("detect.png", inputImg);
   char c = cv::waitKey(5);
   if(c == ' ')
   {
      cv::waitKey(0);
   }
   }
}

