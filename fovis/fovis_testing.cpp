#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#include <cstdio>
#include <ctime>

#include <iostream>
#include <string>
#include <fovis.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "persistence1d.hpp"
#include "zedin.hpp"


using namespace std;
using namespace cv;

bool leftCamera = true;
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

double avgOfMat(Mat &img, Mat &mask) { //this averages mats without counting NaN
	CV_Assert(img.depth() == CV_32F); //must be of type used for depth
	float* ptr_img;
	float* ptr_mask;
	double sum = 0;
	for(int j = 0;j < img.rows;j++){

	    ptr_img = img.ptr<float>(j);
	    ptr_mask = mask.ptr<float>(j);

	    for(int i = 0;i < img.cols;i++){
		if(ptr_img[i] != NAN && ptr_mask[i] == 255) //if the data is not NAN and mask is true
		    sum = sum + ptr_img[i];
	    }
	}
	return sum / (float)(img.rows * img.cols);
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

  int numHistBins = 64;
  int minDepthInt = 0;
  int maxDepthInt = 1000;
  int stddev_weight_int = 100;
  int rangeRangeInt = 100;

  string detectWindowName = "Background Partitioning Parameters";
  namedWindow(detectWindowName);
  createTrackbar ("Standard Deviation Multiplier", detectWindowName, &stddev_weight_int, 4000, NULL);
  createTrackbar ("Histogram Min Depth", detectWindowName, &minDepthInt, 2000, NULL);
  createTrackbar ("Histogram Max Depth", detectWindowName, &maxDepthInt, 2000, NULL);
  createTrackbar ("Number of bins", detectWindowName, &numHistBins, 128, NULL);
  createTrackbar ("Range of Range", detectWindowName, &rangeRangeInt, 1000, NULL);

  float histRange_arr[2];
  float stddev_weight;

  fovis::Rectification rect(rgb_params);
  fovis::VisualOdometry* odom = new fovis::VisualOdometry(&rect, options);

  Mat frame, depthFrame, histMat, displayFrame, depthMask, depthMask_inv, hist_stddev_mat, hist_mean_mat;
  Mat depth_mean_mat_1, depth_stddev_mat_1, depth_mean_mat_2, depth_stddev_mat_2;
  float* depthImageFloat = new float[cap->width * cap->height];
  double histMaxVal;
  Point  histMaxValLoc;
  fovis::DepthImage depthSource(rgb_params, cap->width, cap->height);
  clock_t startTime;
  vector<float> peaks;
  vector<float> histVec;
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

    depthFrame = Mat(cap->height, cap->width, CV_32FC1, depthImageFloat);
    
    depthSource.setDepthImage(depthImageFloat);

    histRange_arr[0] = minDepthInt / 100.0;
    histRange_arr[1] = maxDepthInt / 100.0;
    stddev_weight = stddev_weight_int / 100.0;

    const float* histRange = { histRange_arr };
    bool uniform = true; bool accumulate = false;
    calcHist(&depthFrame,1,0,Mat(),histMat,1,&numHistBins,&histRange, uniform, accumulate); //create a histogram, depth on the x and num pixels on the y
    int hist_img_width = 512; int hist_img_height = 400;
    int hist_bin_img_width = cvRound( (double)hist_img_width / numHistBins);
    float bin_width_m = (histRange_arr[1] - histRange_arr[0]) / (float)numHistBins;

    histMat.at<float>(0) = 0; //first bin includes NaN and gets very large so ignore them
    //meanStdDev(histMat,hist_stddev_mat,hist_mean_mat); //calculate the mean and stddev of the histogram
    //minMaxLoc(histMat,NULL,&histMaxVal,NULL,&histMaxValLoc); //calculate peak in the histogram

    peaks.clear();
    histVec.clear();
    for(int i = 0; i < histMat.cols; i++)
	histVec.push_back(histMat.at<float>(i));

    p1d::Persistence1D p; //use a library called Persistence to find all maximums
    p.RunPersistence(histVec);
    vector< p1d::TPairedExtrema > Extrema;
    p.GetPairedExtrema(Extrema, 10);
    //Print all found pairs - pairs are sorted ascending wrt. persistence.
    for(vector< p1d::TPairedExtrema >::iterator it = Extrema.begin(); it != Extrema.end(); it++)
    {
        cout << "Persistence: " << (*it).Persistence
             << " minimum index: " << (*it).MinIndex
             << " maximum index: " << (*it).MaxIndex
             << std::endl;
	peaks.push_back((*it).MaxIndex);

    }

    Mat histImage(hist_img_height,hist_img_width, CV_8UC1, Scalar(0,0,0));
    normalize(histMat, histMat, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    for( int i = 1; i < numHistBins; i++ )
    {
      //cout << "current hist point: " << hist_bin_img_width*(i-1) << "," << histMat.at<float>(i) << endl;
      line( histImage, Point( hist_bin_img_width*(i-1), hist_img_height - cvRound(histMat.at<float>(i-1)) ) ,
                       Point( hist_bin_img_width*(i), hist_img_height - cvRound(histMat.at<float>(i)) ),
                       Scalar( 255, 255, 255), 4, 8, 0  );
    }
    //line( histImage, Point(0, range_threshhold), Point(hist_img_width, range_threshhold), Scalar(255,255,255),4,8,0);

    imshow("Histogram",histImage);

    frame.copyTo(displayFrame); //make a color copy for display purposes
    cvtColor(displayFrame,displayFrame,CV_BGR2HSV); //convert to hsv
    inRange(depthFrame,Scalar((peaks[peaks.size() - 1] * bin_width_m) - (rangeRangeInt / 1000.0)), Scalar((peaks[peaks.size() - 1] * bin_width_m) + (rangeRangeInt / 1000.0)),depthMask);
    double depth_mean_1 = avgOfMat(depthFrame,depthMask); //find the average of the "background"
    bitwise_not(depthMask,depthMask_inv); //invert the mat
    double depth_mean_2 = avgOfMat(depthFrame,depthMask_inv); //find the average of the "foreground"
    cout << "Mean of majority: " << depth_mean_1 << endl;
    cout << "Mean of minority: " << depth_mean_2 << endl;
    cout << "Mean of chosen: ";
    if(depth_mean_2 > depth_mean_1) { //if the foreground of the image is the farther one than
	bitwise_not(depthMask,depthMask); //invert the mask because we are obviously wrong about that
	cout << depth_mean_2 << endl;
	} else {
	cout << depth_mean_1 << endl;
	}
    add(displayFrame,Scalar(140,0,0), displayFrame, depthMask); //tint the background so we can see which parts are background for display
    cvtColor(displayFrame,displayFrame,CV_HSV2BGR);
    imshow("frame",displayFrame);
    imshow("background mask", depthMask);


    cvtColor(frame,frame,CV_BGR2GRAY);
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
