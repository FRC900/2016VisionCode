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

#include <zedin.hpp>

using namespace std;
using namespace cv;

int main() {
  ZedIn cap;
  fovis::CameraIntrinsicsParameters rgb_params;
  rgb_params.width = cap.width;
  rgb_params.height = cap.height; //get width and height from the camera

  // TODO read these values from the camera somehow, instead of hard-coding it
  rgb_params.fx = 528.49404721; 
  rgb_params.fy = rgb_params.fx;
  rgb_params.cx = width / 2.0;
  rgb_params.cy = height / 2.0;

  // TODO change this later so we can adjust options
  fovis::VisualOdometryOptions options = fovis::VisualOdometry::getDefaultOptions();

  fovis::Rectification rect(rgb_params);
  fovis::VisualOdometry* odom = new fovis::VisualOdometry(&rect, options);

  Mat frame;
  Mat depthMat;
  while(1)
    {
    cap.getNextFrame(frame,false,false);
    cap.getDepth(depthMat);

    cvtColor(frame,frame,CV_BGR2GRAY);

    odom->processFrame(frame,depthMat);
    
    Eigen::Isometry3d motion_estimate = odom->getMotionEstimate(); //estimate motion

    std::cout << isometryToString(cam_to_local) << " " << 
      isometryToString(motion_estimate) << "\n";
    }
}
