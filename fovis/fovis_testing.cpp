#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

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
  ZedIn cap;
  fovis::CameraIntrinsicsParameters rgb_params;
  memset(&rgb_params,0,sizeof(rgb_params));
  rgb_params.width = cap.width;
  rgb_params.height = cap.height; //get width and height from the camera

  rgb_params.fx = cap.getCameraParams(leftCamera).fx;
  rgb_params.fy = cap.getCameraParams(leftCamera).fy;
  rgb_params.cx = cap.getCameraParams(leftCamera).cx; //get camera intrinsic parameters from zed function
  rgb_params.cy = cap.getCameraParams(leftCamera).cy;

  // TODO change this later so we can adjust options
  fovis::VisualOdometryOptions options = fovis::VisualOdometry::getDefaultOptions();
  options["max-pyramid-level"] = "3"; //default 3
  options["feature-search-window"] = "25"; //default 25
  options["use-subpixel-refinement"] = "true"; //default true

  fovis::Rectification rect(rgb_params);
  fovis::VisualOdometry* odom = new fovis::VisualOdometry(&rect, options);

  Mat frame;
  float* depthImageFloat = new float[cap.width * cap.height];
  fovis::DepthImage depthSource(rgb_params, cap.width, cap.height);
  while(1)
    {
    cap.getNextFrame(frame,leftCamera);

    int pixelCounter = 0;
    for(int y = 0; y < cap.height; y++) {
        for(int x = 0; x < cap.width; x++) {
	    depthImageFloat[pixelCounter] = cap.getDepthPoint(x,y);
	    pixelCounter++;
            }
        }
    depthSource.setDepthImage(depthImageFloat);

    cvtColor(frame,frame,CV_BGR2GRAY);
    imshow("frame",frame);
    if(!frame.isContinuous()) {
	cout << "image is not continuous. Cannot continue, exiting now" << endl;
	return -1;
	}
    uint8_t* pt = (uint8_t*)frame.data;

    odom->processFrame(pt, &depthSource);
	
    //if(odom->getReferenceFrame()->getNumDetectedKeypoints() == odom->getTargetFrame()->getNumDetectedKeypoints()) //this might false positive but not sure if that's important
	//cout << "reference frame reset" << endl;
    Eigen::Isometry3d motion_estimate = odom->getMotionEstimate(); //estimate motion
    Eigen::Isometry3d cam_to_local = odom->getPose();

    std::cout << isometryToString(motion_estimate) << endl; //print out the pose
    waitKey(5);
    }
}
