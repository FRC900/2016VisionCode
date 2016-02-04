#include "FovisLocalizer.hpp"

using namespace std;
using namespace cv;

FovisLocalizer::FovisLocalizer(sl::zed::CamParameters input_params,int in_width, int in_height, cv::Mat& initial_frame) 
{
	memset(&_rgb_params,0,sizeof(_rgb_params)); //set params to 0 to be sure

	_rgb_params.width = in_width;
	_rgb_params.height = in_height; //get width and height from the camera
	_im_width = in_width; //save for later
	_im_height = in_height;

	_rgb_params.fx = input_params.fx;
	_rgb_params.fy = input_params.fy;
	_rgb_params.cx = input_params.cx; 
	_rgb_params.cy = input_params.cy;

	_rect = new fovis::Rectification(_rgb_params);

	initial_frame.copyTo(prev);

	reloadFovis();


}

void FovisLocalizer::reloadFovis()
{

	fovis::VisualOdometryOptions options = fovis::VisualOdometry::getDefaultOptions();
	options["max-pyramid-level"] = to_string(fv_param_max_pyr_level);
	options["feature-search-window"] = to_string(fv_param_feature_search_window);
	options["use-subpixel-refinement"] = "true";
	options["feature-window-size"] = to_string(fv_param_feature_window_size);
	options["target-pixels-per-feature"] = to_string(fv_param_target_ppf);

	_odom = new fovis::VisualOdometry(_rect, options);

}

void FovisLocalizer::processFrame(cv::Mat& img_in, cv::Mat& depth_in)
{

	img_in.copyTo(frame);
	depth_in.copyTo(depthFrame);	

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

	int sumPoints = 0;
	for(int i = 0; i < currCorner2.size(); i++) {
		sumPoints = sumPoints + currCorner2[i].size();
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
			if(abs(optical_flow_magnitude[i]) > (flow_arbitrary_outlier_threshold_int / 100.0) * abs(mag_mean) && flow_good_sectors[i] || optical_flow_magnitude[i] == 0) { 
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
	float* ptr_depthFrame;

	int sectors_passed = 0;
	for(int i = 0; i < flow_good_sectors.size(); i++) { //implement the optical flow into the depth data
		if(!flow_good_sectors[i]) { //true if the sector is bad
			Mat sector_submatrix = Mat(depthFrame,Range(flow_rects[i].tl().y,flow_rects[i].br().y), Range(flow_rects[i].tl().x,flow_rects[i].br().x));
			Mat(sector_submatrix.rows,sector_submatrix.cols,CV_32FC1,Scalar(-2.0)).copyTo(sector_submatrix); //copy
			rectangle(frame,flow_rects[i],Scalar(0,0,255),5);
		} else {
			sectors_passed++;		
		}
	}

	int sumNanPixels = 0;
	for(int j = 0;j < depthFrame.rows;j++){ //for each row
		ptr_depthFrame = depthFrame.ptr<float>(j);
		for(int i = 0;i < depthFrame.cols;i++){ //for each pixel in row
			if(ptr_depthFrame[i] <= 0) {
				ptr_depthFrame[i] = NAN; //set to NaN if negative
				sumNanPixels++;
			} else {
				ptr_depthFrame[i] = ptr_depthFrame[i] / 1000.0; //convert to m
			}
		}
	}

	fovis::DepthImage depthSource(_rgb_params, _im_width, _im_height);
	depthSource.setDepthImage((float*)depthFrame.data); //pass the data into fovis

	uint8_t* pt = (uint8_t*)frameGray.data; //cast to unsigned integer and create pointer to data

	_odom->processFrame(pt, &depthSource); //run visual odometry
	Eigen::Isometry3d m = _odom->getMotionEstimate(); //estimate motion

	Eigen::Vector3d xyz = m.translation();
	Eigen::Vector3d rpy = m.rotation().eulerAngles(0, 1, 2);

	_transform.first[0] = xyz(0);
	_transform.first[1] = xyz(1);
	_transform.first[2] = xyz(2);

	_transform.second[0] = rpy(0) * 180/M_PI;
	_transform.second[1] = rpy(1) * 180/M_PI;
	_transform.second[2] = rpy(2) * 180/M_PI;

	for(int i = 0; i < 3; i++) {
		if(_transform.second[i] >= 90) {
			_transform.second[i] = _transform.second[i] - 180;
		} else if(_transform.second[i] <= -90) {
			_transform.second[i] = _transform.second[i] + 180;
		}

	}
}
