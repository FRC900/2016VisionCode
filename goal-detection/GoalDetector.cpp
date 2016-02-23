#include <opencv2/highgui/highgui.hpp>
#include "GoalDetector.hpp"

using namespace std;
using namespace cv;

GoalDetector::GoalDetector(cv::Point2f fov_size, cv::Size frame_size) :
	_goal_shape(3)
{
	_goal_found = false;
	_fov_size = fov_size;
	_frame_size = frame_size;
	_draw = false;
}

void GoalDetector::wrapConfidence(float &confidence)
{
	confidence = confidence > 0.5 ? 1 - confidence : confidence;
}

void GoalDetector::processFrame(Mat& image, const Mat& depth, Rect &bound)
{
	// Use to mask the contour off from the rest of the 
	// image - used when grabbing depth data for the contour
    Mat contour_mask(image.rows, image.cols, CV_8UC1, Scalar(0));

	// Reset goal_found flag for each frame. Set it later if
	// the goal is found
	_goal_found = false;

	// Look for parts the the image which are within the
	// expected bright green color range
    Mat threshold_image;
	if(!generateThreshold(image, threshold_image, _hue_min, _hue_max, _sat_min, _sat_max, _val_min, _val_max))
	{
		_dist_to_goal = -1.0;
		_angle_to_goal = -1.0;
		bound = Rect(0,0,0,0);
		return;
	}

	// find contours in the thresholded image - these will be blobs
	// of green to check later on to see how well they match the
	// expected shape of the goal
    vector<vector<Point> > contours;
    vector<Vec4i>          hierarchy;
    findContours(threshold_image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    float maxConfidence = 0.f;
	//cout << contours.size() << " goalDetect contours found" << endl;

	int best_contour_index;

	//center of mass as a percentage of the object size from top left
	Point2f com_percent_expected(_goal_shape.com().x / _goal_shape.width(),
				     _goal_shape.com().y / _goal_shape.height());

    for (size_t i = 0; i < contours.size(); i++)
    {
	    Rect br = boundingRect(contours[i]);
		contour_mask.setTo(Scalar(0));
	
		drawContours(contour_mask, contours, i, Scalar(255), CV_FILLED); //create a mask on the contour

		bound = boundingRect(contours[i]);
		// get the minimum and maximum depth values in the contour,
		// copy them into individual floats
		pair<float, float> minMax = utils::minOfDepthMat(depth, contour_mask, br, 10);  
		float depth_z_min = minMax.first;
		float depth_z_max = minMax.second;                        

		//make sure the depth exists before doing other things
		if ((depth_z_min < 0.) || (depth_z_max < 0.)) 
			continue;

		// ObjectType computes a ton of useful properties so create 
		// one for what we're looking at
		ObjectType goal_actual(contours[i]);

		if (goal_actual.area() < 200.0)
			continue;
		//create a trackedobject to get x,y,z of the goal
		TrackedObject goal_tracked_obj(0, _goal_shape, br, minMax.second, _fov_size, _frame_size);

		//percentage of the object filled in
		float filledPercentageActual   = goal_actual.area() / goal_actual.boundingArea();
		float filledPercentageExpected = _goal_shape.area() / _goal_shape.boundingArea();

		//center of mass as a percentage of the object size from top left
		Point2f com_percent_actual((goal_actual.com().x - br.tl().x) / goal_actual.width(),
				           (goal_actual.com().y - br.tl().y) / goal_actual.height());

		//width to height ratio
		float actualRatio   = goal_actual.width() / goal_actual.height();
		float expectedRatio = _goal_shape.width() / _goal_shape.height();

		//parameters for the normal distributions
		//values for standard deviation were determined by taking the standard deviation of a bunch of values from the goal
		pair<float,float> height_normal = make_pair(_goal_height, 0.059273877);
		pair<float,float> com_x_normal  = make_pair(com_percent_expected.x, 0.05157222);
		pair<float,float> com_y_normal  = make_pair(com_percent_expected.y, 0.0439207);
		pair<float,float> area_normal   = make_pair(filledPercentageExpected, 0.057567619);
		pair<float,float> ratio_normal  = make_pair(expectedRatio, 0.537392);
		pair<float,float> ideal_area    = make_pair(1.0, 1.0/3.0);

		//confidence is near 0.5 when value is near the mean
		//confidence is small or large when value is not near mean
		float confidence_height     = utils::normalCFD(height_normal, goal_tracked_obj.getPosition().z);
		float confidence_com_x      = utils::normalCFD(com_x_normal,  com_percent_actual.x);
		float confidence_com_y      = utils::normalCFD(com_y_normal,  com_percent_actual.y);
		float confidence_area       = utils::normalCFD(area_normal,   filledPercentageActual);
		float confidence_ratio      = utils::normalCFD(ratio_normal,  actualRatio);
		float confidence_ideal_area = utils::normalCFD(ideal_area,  (float)br.area() / goal_tracked_obj.getScreenPosition(_fov_size, _frame_size).area());

		//when confidence is near 0 set it to 0
		//when confidence is around 0.5, leave it there 
		//when confidence is near 1 set it to 0
		wrapConfidence(confidence_height);
		wrapConfidence(confidence_com_x);
		wrapConfidence(confidence_com_y);
		wrapConfidence(confidence_area);
		wrapConfidence(confidence_ratio);
		wrapConfidence(confidence_ideal_area);
		
		// higher is better
		float confidence = (confidence_height + confidence_com_x + confidence_com_y + confidence_area + confidence_ratio + confidence_ideal_area) / 6.0;

		/*
		cout << "-------------------------------------------" << endl;
		cout << "Contour " << i << endl;
		cout << "confidence_height: " << confidence_height << endl;
		cout << "confidence_com_x: " << confidence_com_x << endl;
		cout << "confidence_com_y: " << confidence_com_y << endl;
		cout << "confidence_area: " << confidence_area << endl;
		cout << "confidence_ratio: " << confidence_ratio << endl;
		cout << "confidence_ideal_area: " << confidence_ideal_area << endl;
		cout << "confidence: " << confidence << endl;		
		cout << "-------------------------------------------" << endl;
		*/

		if(_draw) {
			drawContours(image, contours,i,Scalar(0,0,255),3);
			rectangle(image, br, Scalar(255,0,0), 2);
			putText(image, to_string(confidence),br.tl(), FONT_HERSHEY_PLAIN, 1, Scalar(255,0,0));
			putText(image, to_string(i), br.br(), FONT_HERSHEY_PLAIN,1,Scalar(0,255,0));
		}

		if(confidence > maxConfidence) {
			float h_dist_with_min = hypotf(depth_z_min, _goal_height); //uses pythagorean theorem to determine horizontal distance to goal using minimum
			float h_dist_with_max = hypotf(depth_z_max, _goal_height + _goal_shape.height()); //this one uses maximum
			float h_dist          = (h_dist_with_max + h_dist_with_min) / 2.0;     //average of the two is more accurate
			float goal_to_center_px  = ((float)br.tl().x + ((float)br.width / 2.0)) - ((float)image.cols / 2.0);                                     //number of pixels from center of contour to center of image (e.g. how far off center it is)
			float goal_to_center_deg = _fov_size.x * (goal_to_center_px / (float)image.cols);                                                                           //converts to angle using the field of view
			_dist_to_goal    = h_dist;
			_angle_to_goal   = goal_to_center_deg * (180.0 / M_PI);
			bound = br;

			maxConfidence = confidence;
			best_contour_index = i;
			_goal_found = true;
		}

		/*vector<string> info;
		info.push_back(to_string(confidence_height));
		info.push_back(to_string(confidence_com_x));
		info.push_back(to_string(confidence_com_y));
		info.push_back(to_string(confidence_area));
		info.push_back(to_string(confidence_ratio));
		info.push_back(to_string(confidence));
		info.push_back(to_string(h_dist));
		info.push_back(to_string(goal_to_center_deg));
		info_writer.log(info); */
	}
	if(_goal_found && _draw)
		rectangle(image, boundingRect(contours[best_contour_index]), Scalar(0,255,0), 2);
}


bool GoalDetector::generateThreshold(const Mat& ImageIn, Mat& ImageOut, int H_MIN, int H_MAX, int S_MIN, int S_MAX, int V_MIN, int V_MAX)
{
    Mat ThresholdLocalImage;

    vector<Mat> SplitImage;
    Mat         SplitImageLE;
    Mat         SplitImageGE;

    cvtColor(ImageIn, ThresholdLocalImage, CV_BGR2HSV, 0);
    split(ThresholdLocalImage, SplitImage);
    Vec3i max( H_MAX, S_MAX, V_MAX );
    Vec3i min( H_MIN, S_MIN, V_MIN );
    for (size_t i = 0; i < SplitImage.size(); i++)
    {
        compare(SplitImage[i], min[i], SplitImageGE, cv::CMP_GE);
        compare(SplitImage[i], max[i], SplitImageLE, cv::CMP_LE);
        bitwise_and(SplitImageGE, SplitImageLE, SplitImage[i]);
    }
    bitwise_and(SplitImage[0], SplitImage[1], ImageOut);
    bitwise_and(SplitImage[2], ImageOut, ImageOut);

    Mat erodeElement(getStructuringElement(MORPH_RECT, Size(3, 3)));
    Mat dilateElement(getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
    erode(ImageOut, ImageOut, erodeElement, Point(-1, -1), 2);
    dilate(ImageOut, ImageOut, dilateElement, Point(-1, -1), 2);
    return (countNonZero(ImageOut) != 0);
}
