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

string isometryToString(const Eigen::Isometry3d& m)
{
  char result[80];
  memset(result, 0, sizeof(result));
  Eigen::Vector3d xyz = m.translation();
  Eigen::Vector3d rpy = m.rotation().eulerAngles(0, 1, 2);
  snprintf(result, 79, "%6.2f %6.2f %6.2f %6.2f %6.2f %6.2f", 
      xyz(0), xyz(1), xyz(2), 
      rpy(0) * 180/M_PI, rpy(1) * 180/M_PI, rpy(2) * 180/M_PI);
  return std::string(result);
}


int main() {
  ZedIn cap;
  fovis::CameraIntrinsicsParameters rgb_params;
  rgb_params.width = cap.width;
  rgb_params.height = cap.height; //get width and height from the camera

  // TODO read these values from the camera somehow, instead of hard-coding it
  rgb_params.fx = 528.49404721; 
  rgb_params.fy = rgb_params.fx;
  rgb_params.cx = cap.width / 2.0;
  rgb_params.cy = cap.height / 2.0;

  // TODO change this later so we can adjust options
  fovis::VisualOdometryOptions options = fovis::VisualOdometry::getDefaultOptions();
  options["max-pyramid-level"] = "1";

  fovis::Rectification rect(rgb_params);
  fovis::VisualOdometry* odom = new fovis::VisualOdometry(&rect, options);

  Mat frame;
  float* depthImageFloat = new float[cap.width * cap.height];
  fovis::DepthImage depthSource(rgb_params, cap.width, cap.height);
  while(1)
    {
    cap.getNextFrame(frame,false,false);
    uchar* depthData = cap.getDepthData(false);
    
    uint8_t* depthPt = (uint8_t*)depthData;
    //memset(depthImageFloat, 0, cap.width*cap.height*sizeof(float));
    for(int i = 0; i < cap.width * cap.height; i++) {
        depthImageFloat[i] = (float)depthPt[i] / 1000.0;
        }
    depthSource.setDepthImage(depthImageFloat);

    cvtColor(frame,frame,CV_BGR2GRAY);
    imshow("frame",frame);
    uint8_t* pt = (uint8_t*)frame.data;
    odom->processFrame(pt, &depthSource);

    Eigen::Isometry3d motion_estimate = odom->getMotionEstimate(); //estimate motion
    Eigen::Isometry3d cam_to_local = odom->getPose();

    std::cout << isometryToString(cam_to_local) << " " << 
    isometryToString(motion_estimate) << "\n";
    waitKey(5);
    }
}
