#include <iostream>
#include <fstream>
#include <cstdio>
#include "opencv2/gpu/gpu.hpp"

#include "classifierio.hpp"
#include "objdetect.hpp"
#include "videoin.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

int main(int argc, char **argv)
{
   const string videoName = "../../../Kevin/imageclassifier/bin_videos/Palmetto 2015/cap2_fixed.avi";
   const string cascadeName = "../cascade_training/classifier_bin_x/cascade.xml";
   VideoIn cap(videoName.c_str());
   cap.frameCounter(800);
   CPU_CascadeDetect detector(cascadeName.c_str());
   const int frameCount = 200;

   // Verfiy the detector loaded
   if( !detector.initialized() )
   {
      cerr << "Could not load " << cascadeName << endl;
      cout <<  "0.0" << endl;
      return -2;
   }
   vector<Rect> detectRects;
   Mat frame;
   minDetectSize = cap.width() * 0.06;
   int64 startTick = getTickCount();
   for (int i = 0; i < frameCount; i++)
   {
      cap.getNextFrame(frame, false);
      detector.Detect(frame, detectRects); 
   }
   int64 endTick    = getTickCount();
   int64 totalTicks = endTick - startTick;
   cout << (double)frameCount / ((double)totalTicks / getTickFrequency()) << endl;
   return 0;
}
