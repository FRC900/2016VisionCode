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

bool reload_fovis = true;
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

void inRangeFloat(Mat &img, Mat &mask, float lb, float ub) {
	float* ptr_img;
	float* ptr_mask;
	double sum = 0;
	for(int j = 0;j < img.rows;j++){

		ptr_img = img.ptr<float>(j);
		ptr_mask = mask.ptr<float>(j);

		for(int i = 0;i < img.cols;i++){
			if(!(isnan(ptr_img[i])) && lb < ptr_img[i] < ub) //if the data is not NAN and mask is true
				ptr_mask[i] = 255;
			else
				ptr_mask[i] = 0;
		}
	}


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
			//cout << "value of depth: " << ptr_img[i] << endl;
			//cout << "value of mask: " << ptr_mask[i] << endl;
			if(!(isnan(ptr_img[i])) && ptr_img[i] != 0 && ptr_mask[i] != 0 && !(isnan(ptr_mask[i]))) //if the data is not NAN and mask is true
				sum = sum + ptr_img[i];
		}
	}
	return sum / (float)(img.rows * img.cols);
}

void setReload(int trackbar_value, void *why_is_this_parameter_even_here_please_dont_do_anything_with_it) {
	reload_fovis = true;
}


int main(int argc, char **argv) {

	omp_set_num_threads(numThreads);
	Eigen::setNbThreads(numThreads); //set threads for eigen so that it runs slightly faster
	cout << "Using " << Eigen::nbThreads() << " threads" << endl;

	ZedIn *cap = NULL;
	if(argc == 2) {
		cap = new ZedIn(argv[1]);
		cerr << "Read SVO file" << endl;
	}
	else {
		cap = new ZedIn;
		cerr << "Initialized camera" << endl;
	}

	fovis::CameraIntrinsicsParameters rgb_params;
	memset(&rgb_params,0,sizeof(rgb_params)); //get intrinsic parameters from the camera (stuff like fx,cx,etc)
	rgb_params.width = cap->width();
	rgb_params.height = cap->height(); //get width and height from the camera

	rgb_params.fx = cap->getCameraParams().fx;
	rgb_params.fy = cap->getCameraParams().fy;
	rgb_params.cx = cap->getCameraParams().cx; //get camera intrinsic parameters from zed function
	rgb_params.cy = cap->getCameraParams().cy;

	int fv_param_max_pyr_level = 3;
	int fv_param_feature_search_window = 25;
	int fv_param_feature_window_size = 9; //variables to add to trackbars
	int fv_param_target_ppf = 250;

	int num_optical_flow_sectors_x = 4;
	int num_optical_flow_sectors_y = 4; //optical flow parameters
	int num_optical_flow_points = 2000;
	int flow_arbitrary_outlier_threshold_int = 100;

	string detectWindowName = "Parameters";
	namedWindow(detectWindowName);

	createTrackbar ("Max Pyramid Level", detectWindowName, &fv_param_max_pyr_level, 10, setReload);
	createTrackbar ("Feature Search Window", detectWindowName, &fv_param_feature_search_window, 100, setReload);
	createTrackbar ("Feature Window Size", detectWindowName, &fv_param_feature_window_size, 50, setReload);
	createTrackbar ("Target Pixels per Feature", detectWindowName, &fv_param_target_ppf, 4000, setReload);

	createTrackbar ("Optical Flow Sectors X", detectWindowName, &num_optical_flow_sectors_x, 32, NULL);
	createTrackbar ("Optical Flow Sectors Y", detectWindowName, &num_optical_flow_sectors_y, 32, NULL);
	createTrackbar ("Optical Flow Points", detectWindowName, &num_optical_flow_points, 4000, NULL);
	createTrackbar ("Optical Flow Threshold", detectWindowName, &num_optical_flow_points, 1000, NULL);

	fovis::Rectification rect(rgb_params); //create fovis objects
	fovis::DepthImage depthSource(rgb_params, cap->width(), cap->height());
	fovis::VisualOdometry* odom;

	Mat frame, frameGray, prev, prevGray, depthFrame, displayFrame;

	Mat display_sectors(cap->height(),cap->width(),CV_8UC1);
	display_sectors.setTo(Scalar(0));
	clock_t startTime;

	cap->update();
	cap->getFrame().copyTo(prev); //initialize to a frame so we can use optical flow

	while(1)
	{

		if(reload_fovis) {

			fovis::VisualOdometryOptions options = fovis::VisualOdometry::getDefaultOptions();

			options["max-pyramid-level"] = to_string(fv_param_max_pyr_level);
			options["feature-search-window"] = to_string(fv_param_feature_search_window);
			options["use-subpixel-refinement"] = "true";
			options["feature-window-size"] = to_string(fv_param_feature_window_size);
			options["target-pixels-per-feature"] = to_string(fv_param_target_ppf);

			odom = new fovis::VisualOdometry(&rect, options);

			reload_fovis = false;
		}

		startTime = clock();

		cap->update();
		cap->getFrame().copyTo(frame); //pull frame from zed

		cvtColor(frame,frameGray,CV_BGR2GRAY); //convert to grayscale 
		cvtColor(prev,prevGray,CV_BGR2GRAY);

		int num_optical_flow_sectors = num_optical_flow_sectors_x * num_optical_flow_sectors_y;

		vector<Point2f> prevCorner, currCorner;
		vector< vector<Point2f> > prevCorner2(num_optical_flow_sectors); //holds an array of small optical point flow arrays
		vector< vector<Point2f> > currCorner2(num_optical_flow_sectors);
		vector< Rect > flow_rects;
		vector<uchar> status;
		vector<float> err;

		goodFeaturesToTrack(prevGray, prevCorner, num_optical_flow_points, 0.01, 30);
		calcOpticalFlowPyrLK(prevGray, frameGray, prevCorner, currCorner, status, err); //calculate optical flow

		int flow_sector_size_x = frame.cols / num_optical_flow_sectors_x;
		int flow_sector_size_y = frame.rows / num_optical_flow_sectors_y;

		for(int i = 0; i < num_optical_flow_sectors_x; i++) {
			for(int j = 0; j < num_optical_flow_sectors_y; j++) {
				Rect sector_rect(Point(i * flow_sector_size_x,j * flow_sector_size_y), Point((i+1) * flow_sector_size_x,(j+1) * flow_sector_size_y));
				flow_rects.push_back(sector_rect);
			} //create rects to segment points found into boxes
		}

		// Status is set to true for each point where a match was found.
		// Use only these points for the rest of the calculations
		for(int i = 0; i < flow_rects.size(); i++) { //for each sector
			for(int j = 0; j < prevCorner.size(); j++) { //for each point
				if(flow_rects[i].contains(prevCorner[j]) && status[j]) { //"contains" is a method to check if point is within a sector
					prevCorner2[i].push_back(prevCorner[j]); //add the point array to its repsective array
					currCorner2[i].push_back(currCorner[j]);
				}
			}
		}
		vector< float > optical_flow_magnitude;

		Mat rigid_transform;
		for(int i = 0; i < prevCorner2.size(); i++) { //for each sector calculate the magnitude of the movement
			if(prevCorner2[i].size() >= 3 && currCorner2[i].size() >= 3) { //don't calculate if no points in the sector
				rigid_transform = estimateRigidTransform(prevCorner2[i], currCorner2[i], false);
				optical_flow_magnitude.push_back(norm(rigid_transform,NORM_L2));
			}
		}

		vector< bool > flow_good_sectors(optical_flow_magnitude.size(),true); //mark all sectors initially as good
		cout << "Number of sectors: " << optical_flow_magnitude.size() << endl;

		int num_sectors_left = flow_good_sectors.size();

		while(1) {
			float mag_mean;
			float sum = 0;

			int good_sectors_prev_size = num_sectors_left;

			for(int i = 0; i < flow_good_sectors.size(); i++) {
				sum = sum + optical_flow_magnitude[i]; //calculate the mean
			}
			mag_mean = sum / (float)optical_flow_magnitude.size();

			for(int i = 0; i < flow_good_sectors.size(); i++) { //this loop iterates through the points and checks if they are outside a range. if they are, then they are eliminated and the mean is recalculated
				if(abs(optical_flow_magnitude[i]) > (flow_arbitrary_outlier_threshold_int / 100.0) * abs(mag_mean) && flow_good_sectors[i] == true) { 
					flow_good_sectors[i] = false;
				}
			}

			num_sectors_left = 0;
			for(int i = 0; i < flow_good_sectors.size(); i++) //count number of sectors left
				if(flow_good_sectors[i]) 
					num_sectors_left++;

			if(good_sectors_prev_size == num_sectors_left) //if we failed to eliminate anything end this loop
				break;
		}
		cout << "Number of sectors left: " << num_sectors_left << endl;

		float fNaN = std::numeric_limits<float>::quiet_NaN(); //get a nan
		cap->getDepth().copyTo(depthFrame);
		float* ptr_depthFrame;
		for(int j = 0;j < depthFrame.rows;j++){ //for each row
			ptr_depthFrame = depthFrame.ptr<float>(j);
			for(int i = 0;i < depthFrame.cols;i++){ //for each pixel in row
				if(ptr_depthFrame[i] <= 0)
					ptr_depthFrame[i] = fNaN; //set to NaN if negative
			}
		}

		int sectors_passed = 0;
		for(int i = 0; i < flow_good_sectors.size(); i++) { //implement the optical flow into the depth data
			if(flow_good_sectors[i]) {
				sectors_passed++;
				Mat sector_submatrix = Mat(depthFrame,Range(flow_rects[i].tl().y,flow_rects[i].br().y), Range(flow_rects[i].tl().x,flow_rects[i].br().x));
				Mat(sector_submatrix.rows,sector_submatrix.cols,CV_8UC1,Scalar(fNaN)).copyTo(sector_submatrix); //copy

				Mat display_sector_submatrix = Mat(display_sectors,Range(flow_rects[i].tl().y,flow_rects[i].br().y), Range(flow_rects[i].tl().x,flow_rects[i].br().x));
				Mat(display_sector_submatrix.rows,display_sector_submatrix.cols,CV_8UC1,Scalar(255)).copyTo(display_sector_submatrix); //display
			}
		}
	
		cout << sectors_passed << " sectors passed" << endl;
		depthSource.setDepthImage((float*)depthFrame.data); //pass the data into fovis

		if(!frameGray.isContinuous()) {
			cout << "image is not continuous. Cannot continue, exiting now" << endl;
			return -1;
		}
		uint8_t* pt = (uint8_t*)frameGray.data; //cast to unsigned integer and create pointer to data

		odom->processFrame(pt, &depthSource); //run visual odometry

		imshow("Optical Flow Allowed Areas", display_sectors);
		imshow("Frame",frame);

		if(odom->getChangeReferenceFrames())
			cout << "reference frame reset" << endl;
		Eigen::Isometry3d motion_estimate = odom->getMotionEstimate(); //estimate motion
		Eigen::Isometry3d cam_to_local = odom->getPose();

		prev = frame.clone(); //copy to prev for next iteration
		std::cout << isometryToString(cam_to_local) << endl; //print out the pose
		std::cout << "Took: " << (((double)clock() - startTime) / CLOCKS_PER_SEC) << " seconds" << endl;
		waitKey(5);
	}
}
