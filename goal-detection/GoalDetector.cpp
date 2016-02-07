#include "GoalDetector.hpp"

using namespace std;
using namespace cv;

GoalDetector::GoalDetector()
{
    _goal_shape_contour.push_back(Point(0, 0));
    _goal_shape_contour.push_back(Point(0, 609.6));
    _goal_shape_contour.push_back(Point(50.8, 609.6));
    _goal_shape_contour.push_back(Point(50.8, 50.8)); //describes vision target shape
    _goal_shape_contour.push_back(Point(762, 50.8));  //in mm because m doesn't work
    _goal_shape_contour.push_back(Point(762, 609.6));
    _goal_shape_contour.push_back(Point(812.8, 609.6));
    _goal_shape_contour.push_back(Point(812.8, 0));
    _goal_shape_rect = boundingRect(_goal_shape_contour);
}

bool GoalDetector::processFrame(Mat& image, Mat& depth, Rect &bound)
{
    Mat hsv_image, threshold_image;
    Mat contour_mask(image.rows, image.cols, CV_8UC1, Scalar(0));

    vector<vector<Point> > contours;
    vector<Vec4i>          hierarchy;
    Point2f                center_of_area;
    Rect                   contour_rect;

    cvtColor(image, hsv_image, COLOR_BGR2HSV); //thresholds the h,s,v values of the image to look for green
    if(!generateThreshold(image, threshold_image, _hue_min, _hue_max, _sat_min, _sat_max, _val_min, _val_max))
    {
	_dist_to_goal = -1;
	_angle_to_goal = -1;
	bound = Rect(0,0,0,0);
	return false;
    }
    findContours(threshold_image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
//find contours in the thresholded image

    float max_contour_area = 0;

    for (size_t i = 0; i < contours.size(); i++)
    {
        if (contourArea(contours[i]) > max_contour_area)                                                                                                             //if this contour is the biggest one pick it
		{
			Moments mu = moments(contours[i], false);                   //get the center of area of the contour
			center_of_area = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00); //this could be used possibly for comparison to the center of area of the shape

			bound = boundingRect(contours[i]);                   //bounding box of the target
			
			contour_mask.setTo(Scalar(0));
			drawContours(contour_mask, contours, i, Scalar(255), CV_FILLED); //create a mask on the contour
			std::pair<float, float> minMax = minOfMat(depth, contour_mask, countPixel, bound);                                                                                               //get the minimum and maximum depth values in the contour
			float depth_z_min = minMax.first;
			float depth_z_max = minMax.second;                                                                                         //actually does some averaging
			float h_dist_with_min = hypotf(depth_z_min, _goal_height);                                                             //uses pythagorean theorem to determine horizontal distance to goal using minimum

			float h_dist_with_max = hypotf(depth_z_max, _goal_height + (_goal_shape_rect.height / 1000.0)); //this one uses maximum

			float h_dist          = (h_dist_with_max + h_dist_with_min) / 2.0;                                                                                           //average of the two is more accurate

			float goal_to_center_px  = ((float)bound.tl().x + ((float)bound.width / 2.0)) - ((float)image.cols / 2.0);                                     //number of pixels from center of contour to center of image (e.g. how far off center it is)
			float goal_to_center_deg = _camera_hfov * (goal_to_center_px / (float)image.cols);                                                                           //converts to angle using the field of view
			if (depth_z_min == -1)
			{
			   h_dist = -1;
			}
			_dist_to_goal    = h_dist;
			_angle_to_goal   = goal_to_center_deg * (180.0 / M_PI);
			max_contour_area = contourArea(contours[i]);     //set variables
		}
	}
	return true;
}


pair<float, float> GoalDetector::minOfMat(Mat& img, Mat& mask, bool (*f)(float), Rect bound_rect, int range)
{
    CV_Assert(img.depth() == CV_32F);                                                    //must be of type used for depth
    CV_Assert(mask.depth() == CV_8UC1);
    double min = numeric_limits<float>::max();
    double max = numeric_limits<float>::min();
    int min_loc_x;
    int min_loc_y;
    int max_loc_x;
    int max_loc_y;
    bool found = false;
    for (int j = bound_rect.tl().y; j <= bound_rect.br().y; j++) //for each row
	{
		float *ptr_img  = img.ptr<float>(j);
		uchar *ptr_mask = mask.ptr<uchar>(j);

		for (int i = bound_rect.tl().x; i <= bound_rect.br().x; i++) //for each pixel in row
		{
			if ((ptr_mask[i] == 255) && f(ptr_img[i]))
			{
				found = true;
				if (ptr_img[i] > max)
				{
					max = ptr_img[i];
					max_loc_x = i;
					max_loc_y = j;
				}

				if (ptr_img[i] < min)
				{
					min = ptr_img[i];
					min_loc_x = i;
					min_loc_y = j;
				}
			}
		}
	}
    if(!found)
    {
	return pair<float, float>(-1, -1);
    }
    float sum_min   = 0;
    int num_pix_min = 0;
    for (int j = (min_loc_y - range); j < (min_loc_y + range); j++)
    {
		float *ptr_img  = img.ptr<float>(j);
		uchar *ptr_mask = mask.ptr<uchar>(j);
        for (int i = (min_loc_x - range); i < (min_loc_x + range); i++)
        {
            if ((0 < i) && (i < img.cols) && (0 < j) && (j < img.rows) && (ptr_mask[i] == 255) && f(ptr_img[i]))
            {
                sum_min += ptr_img[i];
                num_pix_min++;
            }
        }
    }
    float sum_max = 0;
    int num_pix_max = 0;
    for (int j = (max_loc_y - range); j < (max_loc_y + range); j++)
    {
		float *ptr_img  = img.ptr<float>(j);
		uchar *ptr_mask = mask.ptr<uchar>(j);
        for (int i = (max_loc_x - range); i < (max_loc_x + range); i++)
        {
            if ((0 < i) && (i < img.cols) && (0 < j) && (j < img.rows) && (ptr_mask[i] == 255) && f(ptr_img[i]))
            {
                sum_max += ptr_img[i];
                num_pix_max++;
            }
        }
    }
    return pair<float, float>(sum_min / (num_pix_min * 1000.), sum_max / (num_pix_max * 1000.));
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
