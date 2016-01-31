#include "GoalDetector.hpp"

GoalDetector::GoalDetector() {

_goal_shape_contour.push_back(Point(0,0));
_goal_shape_contour.push_back(Point(0,609.6));
_goal_shape_contour.push_back(Point(50.8,609.6));
_goal_shape_contour.push_back(Point(50.8,50.8)); //describes vision target shape
_goal_shape_contour.push_back(Point(762,50.8)); //in mm because m doesn't work
_goal_shape_contour.push_back(Point(762,609.6));
_goal_shape_contour.push_back(Point(812.8,609.6));
_goal_shape_contour.push_back(Point(812.8,0));

}

bool GoalDetector::processFrame(Mat &image, Mat &depth) {

Mat hsv_image, threshold_image;
Mat contour_mask(image.rows, image.cols, CV_8UC1, Scalar(0));
vector< vector<Point> > contours;
vector<Vec4i> hierarchy;
Point2f center_of_area;
Rect contour_rect;

cvtColor(image, hsv_image, COLOR_BGR2HSV); //thresholds the h,s,v values of the image to look for green
generateThreshold(image, threshold_image, _hue_min, _hue_max, _sat_min, _sat_max, _val_min, _val_max);

findContours( threshold_image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
//find contours in the thresholded image

float max_contour_area = 0;

for (int i = 0; i < contours.size(); i++) {
	Moments mu = moments( contours[i], false ); //get the center of area of the contour
	center_of_area = Point2f( mu.m10/mu.m00 , mu.m01/mu.m00 ); //this could be used possibly for comparison to the center of area of the shape

	contour_rect = boundingRect(contours[i]); //bounding box of the target

	contour_mask.setTo(Scalar(0));

	drawContours(contour_mask,contours,i,Scalar(255),CV_FILLED); //create a mask on the contour


	float depth_z_min = minOfMat(depth,contour_mask,countPixel); //get the minimum and maximum depth values in the contour
	float depth_z_max = minOfMat(depth,contour_mask,countPixel,true); //actually does some averaging

	float h_dist_with_min = sqrt( (depth_z_min*depth_z_min) - (_goal_height*_goal_height) ); //uses pythagorean theorem to determine horizontal distance to goal using minimum
	float h_dist_with_max = sqrt( (depth_z_max*depth_z_max) - ((_goal_height+contour_rect.height/1000.0)*(_goal_height+contour_rect.height/1000.0))); //this one uses maximum
	float h_dist = (h_dist_with_max + h_dist_with_min) / 2.0; //average of the two is more accurate

	float goal_to_center_px = ((float)contour_rect.tl().x + ((float)contour_rect.width / 2.0)) - ((float)image.cols / 2.0); //number of pixels from center of contour to center of image (e.g. how far off center it is)
	float goal_to_center_deg = _camera_hfov * (goal_to_center_px / (float)image.cols); //converts to angle using the field of view
	
	if(contourArea(contours[i]) > max_contour_area) { //if this contour is the biggest one pick it
		_dist_to_goal = h_dist; 
		_angle_to_goal = goal_to_center_deg * (180.0 / M_PI);
		max_contour_area = contourArea(contours[i]); //set variables		
		}
	}

}



float GoalDetector::minOfMat(Mat &img, Mat &mask, bool (*f)(float), bool max, int range) { //this actually gets min or max

CV_Assert(img.depth() == CV_32F); //must be of type used for depth
CV_Assert(mask.depth() == CV_8UC1);
float* ptr_img;
uchar* ptr_mask;
double min = 100000000;
if(max)
  min = 0;
int minLoc_x;
int minLoc_y;

for(int j = 0;j < img.rows;j++){ //for each row

    ptr_img = img.ptr<float>(j);
    ptr_mask = mask.ptr<uchar>(j);

    for(int i = 0;i < img.cols;i++){ //for each pixel in row
	if(f(ptr_img[i]) && ptr_mask[i] == 255) {
	    if(max) {
	       if(ptr_img[i] > min)
		   min = ptr_img[i];
		   minLoc_x = i;
		   minLoc_y = j;
	    } else {
	       if(ptr_img[i] < min)
		   min = ptr_img[i];
		   minLoc_x = i;
		   minLoc_y = j;
		}
	}
    }
}

int sum = 0;
int numPix = 0;
for(int j = (minLoc_x - range); j < (minLoc_x + range); j++) {
	for(int i = (minLoc_y - range); i < (minLoc_y + range); i++) {

		if( 0 < i < img.cols && 0 < j < img.rows && f(img.at<float>(i,j)) && mask.at<uchar>(i,j) == 255) {
			sum = sum + img.at<float>(i,j);
			numPix++;
			}

		}	
	}

return sum / (numPix * 1000.0);

}



void GoalDetector::generateThreshold(const Mat &ImageIn, Mat &ImageOut, int H_MIN, int H_MAX, int S_MIN, int S_MAX, int V_MIN, int V_MAX) {

Mat ThresholdLocalImage;
vector<Mat> SplitImage;
Mat SplitImageLE;
Mat SplitImageGE;

cvtColor(ImageIn, ThresholdLocalImage, CV_BGR2HSV, 0);
split(ThresholdLocalImage, SplitImage);
int max[3] = {H_MAX, S_MAX, V_MAX};
int min[3] = {H_MIN, S_MIN, V_MIN};
for (size_t i = 0; i < SplitImage.size(); i++)
{
compare(SplitImage[i], min[i], SplitImageGE, cv::CMP_GE);
compare(SplitImage[i], max[i], SplitImageLE, cv::CMP_LE);
bitwise_and(SplitImageGE, SplitImageLE, SplitImage[i]);
}
bitwise_and(SplitImage[0], SplitImage[1], ImageOut);
bitwise_and(SplitImage[2], ImageOut, ImageOut);

Mat erodeElement (getStructuringElement( MORPH_RECT,Size(3,3)));
Mat dilateElement(getStructuringElement( MORPH_ELLIPSE,Size(2,2)));
erode(ImageOut, ImageOut, erodeElement, Point(-1,-1), 2);
dilate(ImageOut, ImageOut, dilateElement, Point(-1,-1), 2);

}
