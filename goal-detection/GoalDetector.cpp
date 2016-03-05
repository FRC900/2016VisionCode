#include <opencv2/highgui/highgui.hpp>
#include "GoalDetector.hpp"

using namespace std;
using namespace cv;

GoalDetector::GoalDetector(cv::Point2f fov_size, cv::Size frame_size, bool gui) :
	_goal_shape(3)
{
	_goal_found = false;
	_fov_size = fov_size;
	_frame_size = frame_size;
	_min_valid_confidence = 0.2;

    _otsu = 1;
	_blue_scale = 30;
	_red_scale  = 100;

	if (gui)
	{
		cv::namedWindow("Goal Detect Adjustments", CV_WINDOW_NORMAL);
		createTrackbar("Otsu","Goal Detect Adjustments", &_otsu, 1);
		createTrackbar("Blue Scale","Goal Detect Adjustments", &_blue_scale, 100);
		createTrackbar("Red Scale","Goal Detect Adjustments", &_red_scale, 100);
	}
}

// Values around 0.5 are good. Values away from that are progressively
// worse.  Wrap stuff above 0.5 around 0.5 so the range 
// of values go from 0 (bad) to 0.5 (good).
void GoalDetector::wrapConfidence(float &confidence)
{
	confidence = confidence > 0.5 ? 1 - confidence : confidence;
}

void GoalDetector::processFrame(const Mat& image, const Mat& depth)
{
	// Use to mask the contour off from the rest of the 
	// image - used when grabbing depth data for the contour
    Mat contour_mask(image.rows, image.cols, CV_8UC1, Scalar(0));

	// Reset goal_found flag for each frame. Set it later if
	// the goal is found
	_goal_found = false;
	_dist_to_goal = -1.0;
	_angle_to_goal = -1.0;
	_goal_rect = Rect();
	_goal_pos  = Point3f();
	_best_contour_index = -1;
	_confidence.clear();
	_contours.clear();

	// Look for parts the the image which are within the
	// expected bright green color range
    Mat threshold_image;
	if (!generateThresholdAddSubtract(image, threshold_image))
		return;

	// find contours in the thresholded image - these will be blobs
	// of green to check later on to see how well they match the
	// expected shape of the goal
    vector<Vec4i>          hierarchy;
    findContours(threshold_image, _contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    float maxConfidence = _min_valid_confidence;


	//center of mass as a percentage of the object size from top left
	Point2f com_percent_expected(_goal_shape.com().x / _goal_shape.width(),
				     _goal_shape.com().y / _goal_shape.height());

    for (size_t i = 0; i < _contours.size(); i++)
    {
		// ObjectType computes a ton of useful properties so create 
		// one for what we're looking at
		ObjectType goal_actual(_contours[i]);

		if (goal_actual.area() < 100.0)
			continue;

	    Rect br(boundingRect(_contours[i]));
		contour_mask.setTo(Scalar(0));
	
		//create a mask on the contour
		drawContours(contour_mask, _contours, i, Scalar(255), CV_FILLED); 

		// get the minimum and maximum depth values in the contour,
		// copy them into individual floats
		pair<float, float> minMax = utils::minOfDepthMat(depth, contour_mask, br, 10);  
		float depth_z_min = minMax.first;
		float depth_z_max = minMax.second;                        

		// If no depth data, calculate it using FOV and height of
		// the target. This isn't perfect but better than nothing
		if ((depth_z_min < 1.) || (depth_z_max < 1.)) 
			depth_z_min = depth_z_max = distanceUsingFOV(br);

		//create a trackedobject to get x,y,z of the goal
		TrackedObject goal_tracked_obj(0, _goal_shape, br, depth_z_max, _fov_size, _frame_size);

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
		//values for standard deviation were determined by 
		//taking the standard deviation of a bunch of values from the goal
		pair<float,float> height_normal = make_pair(_goal_height, 0.259273877);
		pair<float,float> com_x_normal  = make_pair(com_percent_expected.x, 0.075);
		pair<float,float> com_y_normal  = make_pair(com_percent_expected.y, 0.1539207);
		pair<float,float> area_normal   = make_pair(filledPercentageExpected, 0.33);
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

		// Normalize values between 0 and 0.5
		wrapConfidence(confidence_height);
		wrapConfidence(confidence_com_x);
		wrapConfidence(confidence_com_y);
		wrapConfidence(confidence_area);
		wrapConfidence(confidence_ratio);
		wrapConfidence(confidence_ideal_area);
		
		// higher is better
		float confidence = (confidence_height + confidence_com_x + confidence_com_y + confidence_area + confidence_ratio + confidence_ideal_area) / 6.0;
		_confidence.push_back(confidence);

#if 1
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
#endif

		if (confidence > maxConfidence) 
		{
			// This is the best goal found so far. Save a bunch of
			// info about it.
			maxConfidence = confidence; 
			_goal_found    = true;
			_best_contour_index = i;
			_goal_pos      = goal_tracked_obj.getPosition();
			_dist_to_goal  = hypotf(_goal_pos.x, _goal_pos.y);
			_angle_to_goal = atan2f(_goal_pos.x, _goal_pos.y) * 180. / M_PI;
			_goal_rect     = br;
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
}


// We're looking for pixels which are mostly green
// with a little bit of blue - that should match
// the LED reflected color.
// Do this by splitting channels and combining 
// them into one grayscale channel.
// Start with the green value.  Subtract the red
// channel - this will penalize pixels which have red
// in them, which is good since anything with red
// is an area we should be ignoring. Do the same with 
// blue, except multiply the pixel values by a weight 
// < 1. Using this weight will let blue-green pixels
// show up in the output grayscale
bool GoalDetector::generateThresholdAddSubtract(const Mat& imageIn, Mat& imageOut)
{
    vector<Mat> splitImage;
    Mat         bluePlusRed;

    split(imageIn, splitImage);
	addWeighted(splitImage[0], _blue_scale / 100.0, 
			    splitImage[2], _red_scale / 100.0, 0.0,
				bluePlusRed);
	subtract(splitImage[1], bluePlusRed, imageOut);

    Mat erodeElement(getStructuringElement(MORPH_RECT, Size(3, 3)));
    Mat dilateElement(getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
    erode(imageOut, imageOut, erodeElement, Point(-1, -1), 2);
    dilate(imageOut, imageOut, dilateElement, Point(-1, -1), 2);

	// Use one of two options for adaptive thresholding.  This will turn
	// the gray scale image into a binary black and white one, with pixels
	// above some value being forced white and those below forced to black
	if (_otsu)
		threshold(imageOut, imageOut, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	else
		adaptiveThreshold(imageOut, imageOut, 255.0, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
    return (countNonZero(imageOut) != 0);
}

// Use the camera FOV, image size and rect size to 
// estimate distance to a target
float GoalDetector::distanceUsingFOV(const Rect &rect) const
{
	float percent_image = (float)rect.height/ _frame_size.height;
	float size_fov = percent_image * _fov_size.y; //TODO fov size
	return _goal_shape.height() / (2.0 * tan(size_fov / 2.0));
}

float GoalDetector::dist_to_goal(void) const 
{
 	//floor distance to goal in m 
	return _goal_found ? _dist_to_goal : -1.0; 
}
float GoalDetector::angle_to_goal(void) const 
{ 
	//angle robot has to turn to face goal in degrees
	return _goal_found ? _angle_to_goal : -1.0; 
}  
		
// Screen rect bounding the goal
Rect GoalDetector::goal_rect(void) const
{
	return _goal_found ? _goal_rect : Rect(); 
}

// Goal x,y,z position relative to robot
Point3f GoalDetector::goal_pos(void) const
{
	return _goal_found ? _goal_pos : Point3f(); 
}
void GoalDetector::drawOnFrame(Mat &image) const
{
	for (size_t i = 0; i < _contours.size(); i++)
	{
		drawContours(image, _contours, i, Scalar(0,0,255), 3);
		Rect br(boundingRect(_contours[i]));
		rectangle(image, br, Scalar(255,0,0), 2);
		putText(image, to_string(_confidence[i]), br.tl(), FONT_HERSHEY_PLAIN, 1, Scalar(255,0,0));
		putText(image, to_string(i), br.br(), FONT_HERSHEY_PLAIN, 1, Scalar(0,255,0));
	}

	if(_goal_found)
		rectangle(image, boundingRect(_contours[_best_contour_index]), Scalar(0,255,0), 2);
}

