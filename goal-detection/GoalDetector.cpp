#include <opencv2/highgui/highgui.hpp>
#include "GoalDetector.hpp"

using namespace std;
using namespace cv;

GoalDetector::GoalDetector(cv::Point2f fov_size, cv::Size frame_size)
{
	_goal_found = false;
	_fov_size = fov_size;
	_frame_size = frame_size;
}

void GoalDetector::processFrame(const Mat& image, const Mat& depth, Rect &bound)
{
    Mat threshold_image;
    Mat contour_mask(image.rows, image.cols, CV_8UC1, Scalar(0));

    vector<vector<Point> > contours;
    vector<Vec4i>          hierarchy;
    Point2f                center_of_area;
    Rect                   contour_rect;

	_goal_found = false;
	if(!generateThreshold(image, threshold_image, _hue_min, _hue_max, _sat_min, _sat_max, _val_min, _val_max))
	{
		_dist_to_goal = -1.0;
		_angle_to_goal = -1.0;
		bound = Rect(0,0,0,0);
		return;
	}
    findContours(threshold_image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
//find contours in the thresholded image

    float maxConfidence = 0.f;
	int detectCount = 0;

    for (size_t i = 0; i < contours.size(); i++)
    {

		contour_mask.setTo(Scalar(0));
		drawContours(contour_mask, contours, i, Scalar(255), CV_FILLED); //create a mask on the contour
		std::pair<float, float> minMax = utils::minOfDepthMat(depth, contour_mask, bound,10);  //get the minimum and maximum depth values in the contour
		float depth_z_min = minMax.first;
		float depth_z_max = minMax.second;                         //actually does some averaging

		//make sure the depth exists before doing other things
		if(depth_z_min < 0.)
			continue;

		//ObjectType computes a ton of useful properties so create one for what we're looking at and one for what we expect
		ObjectType goal_shape(3);
		ObjectType goal_actual(contours[i]);
		

		//create a trackedobject to get x,y,z of the goal
		TrackedObject goal_tracked_obj(detectCount++, goal_shape, boundingRect(contours[i]), (minMax.first + minMax.second) / 2.0, _fov_size, _frame_size);

		float filledPercentageActual = goal_actual.area() / goal_actual.boundingArea();
		float filledPercentageExpected = goal_shape.area() / goal_shape.boundingArea();

		//center of mass as a percentage of the object size from top left
		Point2f com_percent_actual;
		com_percent_actual.x = goal_actual.com().x / goal_actual.width();
		com_percent_actual.y = goal_actual.com().y / goal_actual.height();

		Point2f com_percent_expected;
		com_percent_expected.x = goal_shape.com().x / goal_shape.width();
		com_percent_expected.y = goal_shape.com().y / goal_shape.height();

		//parameters for the normal distributions
		_height_normal = std::make_pair(1, 0.2);
		_com_x_normal = std::make_pair(1, 0.2);
		_com_y_normal = std::make_pair(1, 0.2);
		_area_normal = std::make_pair(1, 0.2);

		//confidence is near 0.5 when value is near the mean
		//confidence is small or large when value is not near mean
		float confidence_height = utils::normalCFD(_height_normal.first, _height_normal.second, (goal_tracked_obj.getPosition().z / _goal_height));
		float confidence_com_x = utils::normalCFD(_com_x_normal.first, _com_x_normal.second, (com_percent_actual.x / com_percent_expected.x));
		float confidence_com_y = utils::normalCFD(_com_y_normal.first, _com_y_normal.second, (com_percent_actual.y / com_percent_expected.y));
		float confidence_area = utils::normalCFD(_area_normal.first, _area_normal.first, (filledPercentageActual / filledPercentageExpected));

		//when confidence is near 0.5 set it to 1
		//when confidence is near 0 set it to 0
		//when confidence is near 1 set it to 0
		confidence_height = confidence_height > 0.5 ? 1 - confidence_height : confidence_height;
		confidence_com_x = confidence_com_x > 0.5 ? 1 - confidence_com_x : confidence_com_x;
		confidence_com_y = confidence_com_y > 0.5 ? 1 - confidence_com_y : confidence_com_y;
		confidence_area = confidence_area > 0.5 ? 1 - confidence_area : confidence_area;
		
		cout << "confidence_height: " << confidence_height << endl;
		cout << "confidence_com_x: " << confidence_com_x << endl;
		cout << "confidence_com_y: " << confidence_com_y << endl;
		cout << "confidence_area: " << confidence_area << endl;

		//higher is better
		float confidence = (confidence_height + confidence_com_x + confidence_com_y + confidence_area) / 4.0;

		cout << "confidence: " << confidence << endl;

		if(confidence > maxConfidence) {
			float h_dist_with_min = hypotf(depth_z_min, _goal_height); //uses pythagorean theorem to determine horizontal distance to goal using minimum
			float h_dist_with_max = hypotf(depth_z_max, _goal_height + (goal_shape.height())); //this one uses maximum
			float h_dist          = (h_dist_with_max + h_dist_with_min) / 2.0;     //average of the two is more accurate
			float goal_to_center_px  = ((float)bound.tl().x + ((float)bound.width / 2.0)) - ((float)image.cols / 2.0);                                     //number of pixels from center of contour to center of image (e.g. how far off center it is)
			float goal_to_center_deg = _fov_size.x * (goal_to_center_px / (float)image.cols);                                                                           //converts to angle using the field of view
			_dist_to_goal    = h_dist;
			_angle_to_goal   = goal_to_center_deg * (180.0 / M_PI);

			maxConfidence = confidence;
		}
	}
}


bool GoalDetector::generateThreshold(const Mat& ImageIn, Mat& ImageOut, int H_MIN, int H_MAX, int S_MIN, int S_MAX, int V_MIN, int V_MAX)
{
    Mat ThresholdLocalImage;

    vector<Mat> SplitImage;
    Mat         SplitImageLE;
    Mat         SplitImageGE;

    cvtColor(ImageIn, ThresholdLocalImage, CV_BGR2HSV, 0);
    split(ThresholdLocalImage, SplitImage);
    int max[3] = { H_MAX, S_MAX, V_MAX };
    int min[3] = { H_MIN, S_MIN, V_MIN };
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
