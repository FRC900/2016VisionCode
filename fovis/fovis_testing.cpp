#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#include <cstdio>
#include <ctime>

#include <iostream>
#include <string>
#include <fovis/fovis.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "zedin.hpp"

using namespace std;
using namespace cv;

bool leftCamera = false;
int numThreads = 8;


string isometryToString(const Eigen::Isometry3d& m)
{
  char result[80];
  memset(result, 0, sizeof(result));
  Eigen::Vector3d xyz = m.translation();
  Eigen::Vector3d rpy = m.rotation().eulerAngles(0, 1, 2);
  snprintf(result, 79, " X=%6.2f Y=%6.2f Z=%6.2f R=%6.2f P=%6.2f Y=%6.2f", 
      xyz(0), xyz(1), xyz(2), 
      rpy(0) * 180/M_PI, rpy(1) * 180/M_PI, rpy(2) * 180/M_PI);
  return std::string(result);
}


int main() {

  omp_set_num_threads(numThreads);
  Eigen::setNbThreads(numThreads);
  cout << "Using " << Eigen::nbThreads() << " threads" << endl;

  ZedIn cap;
  fovis::CameraIntrinsicsParameters rgb_params;
  memset(&rgb_params,0,sizeof(rgb_params));
  rgb_params.width = cap.width;
  rgb_params.height = cap.height; //get width and height from the camera

  rgb_params.fx = cap.getCameraParams(leftCamera).fx;
  rgb_params.fy = cap.getCameraParams(leftCamera).fy;
  rgb_params.cx = cap.getCameraParams(leftCamera).cx; //get camera intrinsic parameters from zed function
  rgb_params.cy = cap.getCameraParams(leftCamera).cy;

  /*cout << rgb_params.fx << endl;
  cout << rgb_params.fy<< endl;
  cout << rgb_params.cx << endl;
  cout << rgb_params.cy << endl; */


  // TODO change this later so we can adjust options
  fovis::VisualOdometryOptions options = fovis::VisualOdometry::getDefaultOptions();
  options["max-pyramid-level"] = "3"; //default 3
  options["feature-search-window"] = "25"; //default 25
  options["use-subpixel-refinement"] = "true"; //default true
  options["feature-window-size"] = "9"; //default 9
  options["target-pixels-per-feature"] = "250"; //default 250

  fovis::Rectification rect(rgb_params);
  fovis::VisualOdometry* odom = new fovis::VisualOdometry(&rect, options);

  Mat frame;
  Mat depthFrame;
  float* depthImageFloat = new float[cap.width * cap.height];
  fovis::DepthImage depthSource(rgb_params, cap.width, cap.height);
  clock_t startTime;
  while(1)
    {
    startTime = clock();
    cap.getNextFrame(frame,leftCamera);

    int pixelCounter = 0;
    float depthPoint;
    for(int y = 0; y < cap.height; y++) {
        for(int x = 0; x < cap.width; x++) {
	    depthPoint = cap.getDepthPoint(x,y);
	    if(depthPoint <= 0) {
		depthImageFloat[pixelCounter] = NAN;		
	    }
	    else {
		depthImageFloat[pixelCounter] = depthPoint;
	    }
	    pixelCounter++;
            }
        }
    depthFrame = Mat(cap.width, cap.height, CV_32FC1, depthImageFloat);

    depthSource.setDepthImage(depthImageFloat);

    cvtColor(frame,frame,CV_BGR2GRAY);
    imshow("frame",frame);
    if(!frame.isContinuous()) {
	cout << "image is not continuous. Cannot continue, exiting now" << endl;
	return -1;
	}
    uint8_t* pt = (uint8_t*)frame.data;

    odom->processFrame(pt, &depthSource);
	
    if(odom->getChangeReferenceFrames())
	cout << "reference frame reset" << endl;
    Eigen::Isometry3d motion_estimate = odom->getMotionEstimate(); //estimate motion
    Eigen::Isometry3d cam_to_local = odom->getPose();

    std::cout << isometryToString(cam_to_local) << endl; //print out the pose
    std::cout << "Took: " << (((double)clock() - startTime) / CLOCKS_PER_SEC) << " seconds" << endl;
    waitKey(5);
    }
}
