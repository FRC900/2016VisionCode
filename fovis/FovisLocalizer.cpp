#include "FovisLocalizer.hpp"

using namespace std;
using namespace cv;

FovisLocalizer::FovisLocalizer(const sl::zed::CamParameters &input_params, const cv::Mat& initial_frame) :
	_rect(NULL),
	_odom(NULL)
{
	memset(&_rgb_params,0,sizeof(_rgb_params)); //set params to 0 to be sure

	_rgb_params.width = initial_frame.cols;
	_rgb_params.height = initial_frame.rows; //get width and height from the camera

	_rgb_params.fx = input_params.fx;
	_rgb_params.fy = input_params.fy;
	_rgb_params.cx = input_params.cx; 
	_rgb_params.cy = input_params.cy;

	_rect = new fovis::Rectification(_rgb_params);

	cvtColor(initial_frame, prevGray, CV_BGR2GRAY);

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

	if (_odom)
		delete _odom;
	_odom = new fovis::VisualOdometry(_rect, options);
}

FovisLocalizer::~FovisLocalizer()
{
	if (_rect)
		delete _rect;
	if (_odom)
		delete _odom;
}

void FovisLocalizer::processFrame(const cv::Mat& img_in, const cv::Mat& depth_in)
{
	if (depth_in.empty())
		return;
	depthFrame = depth_in.clone();

	cvtColor(img_in, frameGray, CV_BGR2GRAY); // convert to grayscale 

	int num_optical_flow_sectors = num_optical_flow_sectors_x * num_optical_flow_sectors_y;

	vector<Point2f> prevCorner, currCorner;
	vector< vector<Point2f> > prevCorner2(num_optical_flow_sectors); //holds an array of small optical point flow arrays
	vector< vector<Point2f> > currCorner2(num_optical_flow_sectors);
	vector< Rect > flow_rects;
	vector<uchar> status;
	vector<float> err;

	goodFeaturesToTrack(prevGray, prevCorner, num_optical_flow_points, 0.01, 30);
	calcOpticalFlowPyrLK(prevGray, frameGray, prevCorner, currCorner, status, err); //calculate optical flow
	prevGray = frameGray.clone();

	int flow_sector_size_x = img_in.cols / num_optical_flow_sectors_x;
	int flow_sector_size_y = img_in.rows / num_optical_flow_sectors_y;

	for(int i = 0; i < num_optical_flow_sectors_x; i++) {
		for(int j = 0; j < num_optical_flow_sectors_y; j++) {
			Rect sector_rect(Point(i * flow_sector_size_x,j * flow_sector_size_y), Point((i+1) * flow_sector_size_x,(j+1) * flow_sector_size_y));
			flow_rects.push_back(sector_rect);
		} //create rects to segment points found into boxes
	}

	// Status is set to true for each point where a match was found.
	// Use only these points for the rest of the calculations

	for(size_t i = 0; i < flow_rects.size(); i++) { //for each sector
		for(size_t j = 0; j < prevCorner.size(); j++) { //for each point
			if(status[j] && flow_rects[i].contains(prevCorner[j])) { //"contains" is a method to check if point is within a sector
				prevCorner2[i].push_back(prevCorner[j]); //add the point array to its repsective array
				currCorner2[i].push_back(currCorner[j]);
			}
		}
	}

	vector< float > optical_flow_magnitude;

	Mat rigid_transform;
	for(size_t i = 0; i < prevCorner2.size(); i++) { //for each sector calculate the magnitude of the movement
		if(prevCorner2[i].size() >= 3 && currCorner2[i].size() >= 3) { //don't calculate if no points in the sector
			rigid_transform = estimateRigidTransform(prevCorner2[i], currCorner2[i], false);
			optical_flow_magnitude.push_back(norm(rigid_transform,NORM_L2));
		}
	}

	vector< bool > flow_good_sectors(optical_flow_magnitude.size(),true); //mark all sectors initially as good

	size_t num_sectors_left = flow_good_sectors.size();

	while(1) {
		float mag_mean;
		float sum = 0;

		size_t good_sectors_prev_size = num_sectors_left;

		for(size_t i = 0; i < flow_good_sectors.size(); i++) {
			sum += optical_flow_magnitude[i]; //calculate the mean
		}
		mag_mean = sum / (float)optical_flow_magnitude.size();

		//this loop iterates through the points and checks if they 
		//are outside a range. if they are, then they are eliminated 
		//and the mean is recalculated
		for(size_t i = 0; i < flow_good_sectors.size(); i++) { 
			if((optical_flow_magnitude[i]== 0) || 
			   (flow_good_sectors[i] && abs(optical_flow_magnitude[i]) > (flow_arbitrary_outlier_threshold_int / 100.0) * abs(mag_mean))) { 
				flow_good_sectors[i] = false;
			}
		}

		num_sectors_left = 0;
		for(size_t i = 0; i < flow_good_sectors.size(); i++) //count number of sectors left
			if(flow_good_sectors[i]) 
				num_sectors_left++;

		if(good_sectors_prev_size == num_sectors_left) //if we failed to eliminate anything end this loop
			break;
	}
	float* ptr_depthFrame;

	for(size_t i = 0; i < flow_good_sectors.size(); i++) { //implement the optical flow into the depth data
		if(!flow_good_sectors[i]) { //true if the sector is bad
			Mat sector_submatrix = Mat(depthFrame,Range(flow_rects[i].tl().y,flow_rects[i].br().y), Range(flow_rects[i].tl().x,flow_rects[i].br().x));
			Mat(sector_submatrix.rows,sector_submatrix.cols,CV_32FC1,Scalar(-2.0)).copyTo(sector_submatrix); //copy
			cout << "Sector " << i << " bad" << endl;
		}
	}

	for(int j = 0;j < depthFrame.rows;j++){ //for each row
		ptr_depthFrame = depthFrame.ptr<float>(j);
		for(int i = 0;i < depthFrame.cols;i++){ //for each pixel in row
			if(ptr_depthFrame[i] <= 0) {
				ptr_depthFrame[i] = NAN; //set to NaN if negative
			} else {
				ptr_depthFrame[i] /= 1000.0; //convert to m
			}
		}
	}

	fovis::DepthImage depthSource(_rgb_params, frameGray.cols, frameGray.rows);
	cout << "params.fx = " << _rgb_params.fx << endl;
	cout << "params.fy = " << _rgb_params.fy << endl;
	cout << "params.cx = " << _rgb_params.cx << endl;
	cout << "params.cy = " << _rgb_params.cy << endl;

	depthSource.setDepthImage((float*)depthFrame.data); //pass the data into fovis

	uint8_t* pt = (uint8_t*)frameGray.data; //cast to unsigned integer and create pointer to data

	_odom->processFrame(pt, &depthSource); //run visual odometry
	Eigen::Isometry3d m = _odom->getMotionEstimate(); //estimate motion
	_transform_eigen = m;

	Eigen::Vector3d xyz = m.translation();
	Eigen::Vector3d rpy = m.rotation().eulerAngles(0, 1, 2);

	_transform.first[0] = xyz(0);
	_transform.first[1] = xyz(1);
	_transform.first[2] = xyz(2);

	_transform.second[0] = rpy(0) * 180/M_PI;
	_transform.second[1] = rpy(1) * 180/M_PI;
	_transform.second[2] = rpy(2) * 180/M_PI;

	cout << "transform " << _transform.first << " " << _transform.second << endl;
	for(int i = 0; i < 3; i++) {
		if(_transform.second[i] >= 90) {
			_transform.second[i] = _transform.second[i] - 180;
		} else if(_transform.second[i] <= -90) {
			_transform.second[i] = _transform.second[i] + 180;
		}
	}
}
