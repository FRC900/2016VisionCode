#include <opencv2/highgui/highgui.hpp>
#include "GoalDetector.hpp"

using namespace std;
using namespace cv;

//#define VERBOSE

GoalDetector::GoalDetector(cv::Point2f fov_size, cv::Size frame_size, bool gui) :
	_goal_shape(3),
	_fov_size(fov_size),
	_frame_size(frame_size),
	_isValid(false),
	_pastRects(2),
	_min_valid_confidence(0.25),
	_otsu_threshold(8.),
	_blue_scale(30),
	_red_scale(60)
{
	if (gui)
	{
		cv::namedWindow("Goal Detect Adjustments", CV_WINDOW_NORMAL);
		createTrackbar("Blue Scale","Goal Detect Adjustments", &_blue_scale, 100);
		createTrackbar("Red Scale","Goal Detect Adjustments", &_red_scale, 100);
		createTrackbar("Otsu Threshold","Goal Detect Adjustments", &_otsu_threshold, 255);
	}
}

// Compute a confidence score for an actual measurement given
// the expected value and stddev of that measurement
// Values around 0.5 are good. Values away from that are progressively
// worse.  Wrap stuff above 0.5 around 0.5 so the range 
// of values go from 0 (bad) to 0.5 (good).
float GoalDetector::createConfidence(float expectedVal, float expectedStddev, float actualVal)
{
	pair<float,float> expectedNormal(expectedVal, expectedStddev);
	float confidence = utils::normalCFD(expectedNormal, actualVal);
	return confidence > 0.5 ? 1 - confidence : confidence;
}

void GoalDetector::processFrame(const Mat& image, const Mat& depth)
{
	// Use to mask the contour off from the rest of the 
	// image - used when grabbing depth data for the contour
    Mat contour_mask(image.rows, image.cols, CV_8UC1, Scalar(0));

	// Reset previous detection vars
	_isValid = false;
	_dist_to_goal = -1.0;
	_angle_to_goal = -1.0;
	_goal_rect = Rect();
	_goal_pos  = Point3f();
	_confidence.clear();
	_contours.clear();

	// Look for parts the the image which are within the
	// expected bright green color range
    Mat threshold_image;
	if (!generateThresholdAddSubtract(image, threshold_image))
	{
		_pastRects.push_back(SmartRect(Rect()));
		return;
	}

	// find contours in the thresholded image - these will be blobs
	// of green to check later on to see how well they match the
	// expected shape of the goal
    vector<Vec4i>          hierarchy;
    findContours(threshold_image, _contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Initialize maxConfidence to the min confidence we need
	// to think it is a goal. This way we'll only use the goal
	// with the highest confidence if its confidence score is
	// above the minimum
    float maxConfidence = _min_valid_confidence;

	// Create some target stats based on our idealized goal model
	//center of mass as a percentage of the object size from top left
	const Point2f com_percent_expected(_goal_shape.com().x / _goal_shape.width(),
									   _goal_shape.com().y / _goal_shape.height());
	// Ratio of contour area to bounding box area
	const float filledPercentageExpected = _goal_shape.area() / _goal_shape.boundingArea();

	// Aspect ratio of the goal
	const float expectedRatio = _goal_shape.width() / _goal_shape.height();

    for (size_t i = 0; i < _contours.size(); i++)
    {
		// ObjectType computes a ton of useful properties so create 
		// one for what we're looking at
	    Rect br(boundingRect(_contours[i]));

		// Remove objects which are obviously too small
		// TODO :: Tune me, make me a percentage of screen area?
		if ((br.area() < 450.0) || (br.area() > 8500))
		{
#ifdef VERBOSE
			cout << "Contour " << i << " area out of range " << br.area() << endl;
#endif
			_confidence.push_back(0);
			continue;
		}

		// Remove objects too low on the screen - these can't
		// be goals. Should stop the robot from admiring its
		// reflection in the diamond-plate at the end of the field
		if (br.br().y > (image.rows * (2./3)))
		{ 
#ifdef VERBOSE
			cout << "Contour " << i << " br().y out of range "<< br.br().y << endl;
#endif
			_confidence.push_back(0);
			continue;
		}

	
		//create a mask which is the same shape as the contour
		contour_mask.setTo(Scalar(0));
		drawContours(contour_mask, _contours, i, Scalar(255), CV_FILLED); 

		// get the minimum and maximum depth values in the contour,
		// copy them into individual floats
		pair<float, float> minMax = utils::minOfDepthMat(depth, contour_mask, br, 10);  
		float depth_z_min = minMax.first;
		float depth_z_max = minMax.second;                        

		// If no depth data, calculate it using FOV and height of
		// the target. This isn't perfect but better than nothing
		if ((depth_z_min <= 0.) || (depth_z_max <= 0.)) 
			depth_z_min = depth_z_max = distanceUsingFOV(br);

		// TODO : Figure out how well this works in practice
		// Filter out goals which are too close or too far
		if ((depth_z_min < 1.) || (depth_z_max > 12.))
		{
#ifdef VERBOSE
			cout << "Contour " << i << " depth out of range "<< depth_z_min << " / " << depth_z_max << endl;
#endif
			_confidence.push_back(0);
			continue;
		}

		// Since the goal is a U shape, there should be bright pixels
		// at the bottom center of the contour and dimmer ones in the 
		// middle going towards the top. Check for that here
		Mat topMidCol(threshold_image(Rect(cvRound(br.tl().x + br.width / 2.), br.tl().y, 
						                   1, cvRound(br.height / 2.))));
		Mat botMidCol(threshold_image(Rect(cvRound(br.tl().x + br.width / 2.), cvRound(br.tl().y + 2./3*br.height), 1, cvRound(br.height / 3.))));
		double topMaxVal;
		minMaxLoc(topMidCol, NULL, &topMaxVal);
		double botMaxVal;
		minMaxLoc(botMidCol, NULL, &botMaxVal);
		// The max pixel value in the bottom rows of the
		// middle column should be significantly higher than the
		// max pixel value in the top rows of the middle column
		if (topMaxVal > (.5 * botMaxVal))
		{
#ifdef VERBOSE
			cout << "Contour " << i << " max top middle column val too large "<< topMaxVal << " / " << (botMaxVal *.5) << endl;
#endif
			_confidence.push_back(0);
			continue;
		}
		Mat leftMidRow(threshold_image(Rect(br.tl().x, cvRound(br.tl().y + br.height / 2.), cvRound(br.width / 3.), 1)));
		Mat rightMidRow(threshold_image(Rect(br.tl().x + cvRound(br.width * 2. / 3.), cvRound(br.tl().y + br.height / 2.), cvRound(br.width / 3.), 1))); 
		Mat centerMidRow(threshold_image(Rect(br.tl().x + cvRound(br.width * 3./ 8.), cvRound(br.tl().y + br.height / 2.), cvRound(br.width / 4.), 1)));
		double rightMaxVal;
		minMaxLoc(rightMidRow, NULL, &rightMaxVal);
		double leftMaxVal;
		minMaxLoc(leftMidRow, NULL, &leftMaxVal);
		double centerMaxVal;
		minMaxLoc(centerMidRow, NULL, &centerMaxVal);
		if ((abs(rightMaxVal / leftMaxVal - 1) > .3) || (min(rightMaxVal, leftMaxVal) < 2 * centerMaxVal))
		{
#ifdef VERBOSE
			cout << "Contour " << i << " max center middle row val too large " << centerMaxVal * 2. << " / " << min(rightMaxVal, leftMaxVal) << endl;
			cout << "Right: " << rightMidRow << ", Left: " << leftMidRow << endl;
#endif
			_confidence.push_back(0);
			continue;
		}
		//create a trackedobject to get various statistics
		//including area and x,y,z position of the goal
		ObjectType goal_actual(_contours[i]);
		TrackedObject goal_tracked_obj(0, _goal_shape, br, depth_z_max, _fov_size, _frame_size, -9.5 * M_PI / 180.0);

		//percentage of the object filled in
		float filledPercentageActual   = goal_actual.area() / goal_actual.boundingArea();

		//center of mass as a percentage of the object size from top left
		Point2f com_percent_actual((goal_actual.com().x - br.tl().x) / goal_actual.width(),
				                   (goal_actual.com().y - br.tl().y) / goal_actual.height());

		//width to height ratio
		float actualRatio = goal_actual.width() / goal_actual.height();

		// Gets the bounding box area observed divided by the
		// bounding box area calculated given goal size and distance
		// For an object the size of a goal we'd expect this to be 
		// close to 1.0 with some variance due to perspective
		float actualScreenArea = (float)br.area() / goal_tracked_obj.getScreenPosition(_fov_size, _frame_size).area();

		//parameters for the normal distributions
		//values for standard deviation were determined by 
		//taking the standard deviation of a bunch of values from the goal
		//confidence is near 0.5 when value is near the mean
		//confidence is small or large when value is not near mean
		float confidence_height      = createConfidence(_goal_height, 0.259273877, goal_tracked_obj.getPosition().z - _goal_shape.height() / 2.0);
		float confidence_com_x       = createConfidence(com_percent_expected.x, 0.075,  com_percent_actual.x);
		float confidence_com_y       = createConfidence(com_percent_expected.y, 0.1539207,  com_percent_actual.y);
		float confidence_filled_area = createConfidence(filledPercentageExpected, 0.33,   filledPercentageActual);
		float confidence_ratio       = createConfidence(expectedRatio, 0.537392,  actualRatio);
		float confidence_screen_area = createConfidence(1.0, 0.5,  actualScreenArea);
		
		// higher is better
		float confidence = (confidence_height + confidence_com_x + confidence_com_y + confidence_filled_area + confidence_ratio/2. + confidence_screen_area/2.) / 5.0;
		_confidence.push_back(confidence);

#ifdef VERBOSE
		cout << "-------------------------------------------" << endl;
		cout << "Contour " << i << endl;
		cout << "confidence_height: " << confidence_height << endl;
		cout << "confidence_com_x: " << confidence_com_x << endl;
		cout << "confidence_com_y: " << confidence_com_y << endl;
		cout << "confidence_filled_area: " << confidence_filled_area << endl;
		cout << "confidence_ratio: " << confidence_ratio << endl;
		cout << "confidence_screen_area: " << confidence_screen_area << endl;
		cout << "confidence: " << confidence << endl;		
		cout << "br.area() " << br.area() << endl;
		cout << "br.br().y " << br.br().y << endl;
		cout << "Max middle row "<< topMaxVal << " / " << (botMaxVal *.5) << endl;
		cout << "-------------------------------------------" << endl;
#endif

		if (confidence > maxConfidence) 
		{
			// This is the best goal found so far. Save a bunch of
			// info about it.
			maxConfidence  = confidence;
			_isValid       = true;
			_goal_pos      = goal_tracked_obj.getPosition();
			_dist_to_goal  = hypotf(_goal_pos.x, _goal_pos.y);
			_angle_to_goal = atan2f(_goal_pos.x, _goal_pos.y) * 180. / M_PI;
			_goal_rect     = br;
		}

		/*vector<string> info;
		info.push_back(to_string(confidence_height));
		info.push_back(to_string(confidence_com_x));
		info.push_back(to_string(confidence_com_y));
		info.push_back(to_string(confidence_filled_area));
		info.push_back(to_string(confidence_ratio));
		info.push_back(to_string(confidence));
		info.push_back(to_string(h_dist));
		info.push_back(to_string(goal_to_center_deg));
		info_writer.log(info); */
	}
	_pastRects.push_back(SmartRect(_goal_rect));
	isValid();
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
    Mat dilateElement(getStructuringElement(MORPH_RECT, Size(3, 3)));
	for (int i = 0; i < 2; ++i)
	{
		erode(imageOut, imageOut, erodeElement, Point(-1, -1), 1);
		dilate(imageOut, imageOut, dilateElement, Point(-1, -1), 1);
	}

	// Use Ostu adaptive thresholding.  This will turn
	// the gray scale image into a binary black and white one, with pixels
	// above some value being forced white and those below forced to black
	// The value to used as the split between black and white is returned
	// from the function.  If this value is too low, it means the image is
	// really dark and the returned threshold image will be mostly noise.
	// In that case, skip processing it entirely.
	double otsuThreshold = threshold(imageOut, imageOut, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
#ifdef VERBOSE
	cout << "OSTU THRESHOLD " << otsuThreshold << endl;
#endif
	if (otsuThreshold < _otsu_threshold)
		return false;
    return (countNonZero(imageOut) != 0);
}

// Use the camera FOV, image size and rect size to 
// estimate distance to a target
float GoalDetector::distanceUsingFOV(const Rect &rect) const
{
	float percent_image = (float)rect.height / _frame_size.height;
	float size_fov = percent_image * _fov_size.y; //TODO fov size
	return _goal_shape.height() / (2.0 * tanf(size_fov / 2.0));
}

float GoalDetector::dist_to_goal(void) const 
{
 	//floor distance to goal in m 
	return _isValid ? _dist_to_goal : -1.0; 
}

float GoalDetector::angle_to_goal(void) const 
{ 
	//angle robot has to turn to face goal in degrees
	if (!_isValid)
		return -1;

	float mag = fabsf(_angle_to_goal);
	float delta = 0;
	if (mag >= 40)
		delta = 2.0;
	else if (mag >= 35)
		delta = 1.5;
	else if (mag >= 30) 
		delta = 1.0;
	else if (mag >= 25) 
		delta = 0.5;

	//cout << "angle " << _angle_to_goal << "Mag " << mag << " delta " << delta << " foo " << ((_angle_to_goal < 0) ? delta : -delta) << endl;
	//return _isValid ? _angle_to_goal : -1.0; 

	return _angle_to_goal + ((_angle_to_goal < 0) ? delta : -delta);
}  
		
// Screen rect bounding the goal
Rect GoalDetector::goal_rect(void) const
{
	return _isValid ? _goal_rect : Rect(); 
}

// Goal x,y,z position relative to robot
Point3f GoalDetector::goal_pos(void) const
{
	return _isValid ? _goal_pos : Point3f(); 
}

// Draw debugging info on frame - all non-filtered contours
// plus their confidence. Highlight the best bounding rect in
// a different color
void GoalDetector::drawOnFrame(Mat &image) const
{
	for (size_t i = 0; i < _contours.size(); i++)
	{
		drawContours(image, _contours, i, Scalar(0,0,255), 3);
		Rect br(boundingRect(_contours[i]));
		rectangle(image, br, Scalar(255,0,0), 2);
		stringstream confStr;
		confStr << fixed << setprecision(2) << _confidence[i];
		putText(image, confStr.str(), br.tl(), FONT_HERSHEY_PLAIN, 1, Scalar(255,0,0));
		putText(image, to_string(i), br.br(), FONT_HERSHEY_PLAIN, 1, Scalar(0,255,0));
	}

	if(!(_pastRects[_pastRects.size() - 1] == SmartRect(Rect())))
		rectangle(image, _pastRects[_pastRects.size() - 1].myRect, Scalar(0,255,0), 2);
}

// Look for the N most recent detected rectangles to be
// the same before returning them as valid. This makes sure
// the camera has stopped moving and has settled
// TODO : See if we want to return different values for
// several frames which have detected goals but at different
// locations vs. several frames which have no detection at all
// in them?
void GoalDetector::isValid()
{
	SmartRect currentRect = _pastRects[0];
	for(auto it = _pastRects.begin() + 1; it != _pastRects.end(); ++it)
	{
		if(!(*it == currentRect))
		{
			_isValid = false;
			return;
		}
	}
	_isValid = true;
}

// Simple class to encapsulate a rect plus a slightly
// more complex than normal comparison function
SmartRect::SmartRect(const cv::Rect &rectangle):
   myRect(rectangle)
{
}

// Overload equals. Make it so that empty rects never
// compare true, even to each other. 
// Also, allow some minor differences in rectangle positions
// to still count as equivalent.
bool SmartRect::operator== (const SmartRect &thatRect)const
{
	if(myRect == Rect() || thatRect.myRect == Rect())
	{
		return false;
	} 
	double intersectArea = (myRect & thatRect.myRect).area();
	double unionArea     = myRect.area() + thatRect.myRect.area() - intersectArea;

	return ((intersectArea / unionArea) >= .8);
}
