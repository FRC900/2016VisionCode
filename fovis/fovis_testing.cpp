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


int main(int argc, char **argv) {

  omp_set_num_threads(numThreads);
  Eigen::setNbThreads(numThreads);
  cout << "Using " << Eigen::nbThreads() << " threads" << endl;
  ZedIn *cap = NULL;
  if(argc == 2) {
  	cap = new ZedIn(argv[1]);
	cout << "Read SVO file" << endl;
  }
  else {
	cap = new ZedIn;
	cout << "Initialized camera" << endl;
  }
  fovis::CameraIntrinsicsParameters rgb_params;
  memset(&rgb_params,0,sizeof(rgb_params));
  rgb_params.width = cap->width;
  rgb_params.height = cap->height; //get width and height from the camera

  rgb_params.fx = cap->getCameraParams(leftCamera).fx;
  rgb_params.fy = cap->getCameraParams(leftCamera).fy;
  rgb_params.cx = cap->getCameraParams(leftCamera).cx; //get camera intrinsic parameters from zed function
  rgb_params.cy = cap->getCameraParams(leftCamera).cy;

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

  Mat frame, depthFrame, histMat;
  float* depthImageFloat = new float[cap->width * cap->height];
  fovis::DepthImage depthSource(rgb_params, cap->width, cap->height);
  clock_t startTime;
  vector<int> ranges;
  while(1)
    {
    startTime = clock();
    cap->getNextFrame(frame,leftCamera);

    int pixelCounter = 0;
    float depthPoint;
    for(int y = 0; y < cap->height; y++) {
        for(int x = 0; x < cap->width; x++) {
	    depthPoint = cap->getDepthPoint(x,y);
	    if(depthPoint <= 0) {
		depthImageFloat[pixelCounter] = NAN;		
	    }
	    else {
		depthImageFloat[pixelCounter] = depthPoint;
	    }
	    pixelCounter++;
            }
        }

    depthFrame = Mat(cap->width, cap->height, CV_32FC1, depthImageFloat);
    
    depthSource.setDepthImage(depthImageFloat);

    float histRange_arr[] = {0,10};
    const float* histRange = { histRange_arr };
    int numHistBins = 128;
    bool uniform = true; bool accumulate = false;
    calcHist(&depthFrame,1,0,Mat(),histMat,1,&numHistBins,&histRange, uniform, accumulate); //create a histogram, depth on the x and num pixels on the y
    int hist_img_width = 512; int hist_img_height = 400;
    int hist_bin_img_width = cvRound( (double)hist_img_width / numHistBins);
    //cout << "histogram bin width: " << ((histRange_arr[1] - histRange_arr[0]) / (float)numHistBins) * 100.0 << " cm" << endl;
    float bin_width_m = (histRange_arr[1] - histRange_arr[0]) / (float)numHistBins;

    float stddev_weight = 2; //how many standard deviations away it needs to be to be considered a peak

    Mat hist_stddev_mat, hist_mean_mat; //output of meanStdDev
    histMat.at<float>(0) = 0; //first bin includes NaN and gets very large so ignore them
    meanStdDev(histMat,hist_stddev_mat,hist_mean_mat); //calculate the mean and stddev of the histogram
    ranges.clear();
    double range_threshhold = hist_stddev_mat.at<double>(0) * stddev_weight + hist_mean_mat.at<double>(0); //if a point is above this threshhold we consider it a peak
    for(int i = 0; i < histMat.rows * histMat.cols; i++) { //this creates ranges of depths that are peaks
	if(histMat.at<int>(i) > range_threshhold) //if the point is on a range
	    if(!(histMat.at<int>(i-1) > range_threshhold)) //if the previous point is not on a range
		ranges.push_back(i);
	    else if(!(histMat.at<int>(i+1) > range_threshhold)) //or if the next point is not on a range
		ranges.push_back(i);
	}

    for(int i = 0; i < ranges.size(); i+=2) { //range * bin width = depth
	cout << "Range " << i << " : " << ranges[i] * bin_width_m << "-" << ranges[i+1] * bin_width_m << endl;	
	}
    //cout << "Number of ranges: " << ranges.size() << endl;
    //cout << "Last range in m: " << ranges[ranges.size() - 2] * bin_width_m << " to " << ranges[ranges.size() - 1] * bin_width_m << endl;

    Mat histImage(hist_img_height,hist_img_width, CV_8UC1, Scalar(0,0,0));
    for(int i = 0; i < histMat.rows * histMat.cols; i++) {
	if(histMat.at<int>(i) < range_threshhold) //if not a peak set to zero, so that we can only see peaks
	    histMat.at<int>(i) = 0;
	}
    normalize(histMat, histMat, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    for( int i = 1; i < numHistBins; i++ )
    {
      //cout << "current hist point: " << hist_bin_img_width*(i-1) << "," << histMat.at<float>(i) << endl;
      line( histImage, Point( hist_bin_img_width*(i-1), hist_img_height - cvRound(histMat.at<float>(i-1)) ) ,
                       Point( hist_bin_img_width*(i), hist_img_height - cvRound(histMat.at<float>(i)) ),
                       Scalar( 255, 255, 255), 4, 8, 0  );
    }
    imshow("Histogram",histImage);

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
