#include <iostream>
#include <opencv2/opencv.hpp>

#include "zedin.hpp"

using namespace cv;
using namespace std;

void createHistogramImage(const Mat &inputFrame, Mat &histImage)
{
   int histSize = 256;
   float range[] = { 0, 256 } ;
   const float* histRange = { range };
   bool uniform = true, accumulate = false;

   // Split into individual B,G,R channels so we can run a histogram on each
   vector<Mat> splitFrame;
   split (inputFrame, splitFrame);

   Mat hist[3];

   for (size_t i = 0; i < 3; i++) // Compute the histograms:
      calcHist(&splitFrame[i], 1, 0, Mat(), hist[i], 1, &histSize, &histRange, uniform, accumulate );

   // Draw the histograms for B, G and R
   const int hist_w = 512; 
   const int hist_h = 400;
   int bin_w = cvRound( (double) hist_w/histSize );

   histImage = Mat( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
   /// Normalize the result to [ 0, histImage.rows ]
   for (size_t i = 0; i < 3; i++)
      normalize(hist[i], hist[i], 0, histImage.rows, NORM_MINMAX, -1, Mat() );

   const Scalar colors[3] = {Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255)};
   // For each point in the histogram
   for( int ii = 1; ii < histSize; ii++ )
      for (size_t jj = 0; jj < 3; jj++) // Draw for each channel
	 line( histImage, Point( bin_w*(ii-1), hist_h - cvRound(hist[jj].at<float>(ii-1)) ) ,
	       Point( bin_w*(ii), hist_h - cvRound(hist[jj].at<float>(ii)) ),
	       colors[jj], 2, 8, 0  );
}

// Take an input image. Threshold it so that pixels within
// the HSV range specified by [HSV]_[MIN,MAX] are set to non-zero
// and the rest of the image is set to zero. Apply a morph
// open to the resulting image
static void generateThreshold(const Mat &ImageIn, Mat &ImageOut,
	      int H_MIN, int H_MAX, int S_MIN, int S_MAX, int V_MIN, int V_MAX)
{
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

double avgOfMat(Mat &img, Mat &mask, bool (*f)(float) ) { //this averages mats without counting NaN
	CV_Assert(img.depth() == CV_32F); //must be of type used for depth
	CV_Assert(mask.depth() == CV_8UC1);
	float* ptr_img;
	int* ptr_mask;
	double sum = 0;
	double numPix = 0;

	for(int j = 0;j < img.rows;j++){ //for each row

	    ptr_img = img.ptr<float>(j);
	    ptr_mask = mask.ptr<int>(j);

	    for(int i = 0;i < img.cols;i++){ //for each pixel in row
		if(f(ptr_img[i]) && ptr_mask[i] == 255) {
		    sum = sum + ptr_img[i];
		    numPix++;
			}
	    }
	}
	return sum / (numPix * 1000.0);
}

double minOfMat(Mat &img, Mat &mask, bool (*f)(float), bool max=false, int range=10) { //this actually gets min or max
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

bool countPixel(float v) { if( isnan(v) || v <= 0) { return false; } else { return true; } }


int H_MIN = 60; //60-95 is a good range for bright green
int H_MAX = 95;
int S_MIN =  180;
int S_MAX = 255;
int V_MIN =  67;
int V_MAX = 255;

double goal_height = 0.2286;

RNG rng(12345);



int main(int argc, char **argv)
{

   ZedIn *cap = NULL;
   if(argc == 2) {
  	cap = new ZedIn(argv[1]);
	cerr << "Read SVO file" << endl;
   }
   else {
	cap = new ZedIn;
	cerr << "Initialized camera" << endl;
   }

   string trackbarWindowName = "HSV Controls";
   namedWindow(trackbarWindowName, WINDOW_AUTOSIZE);
   createTrackbar( "H_MIN", trackbarWindowName, &H_MIN, 179, NULL);
   createTrackbar( "H_MAX", trackbarWindowName, &H_MAX, 179, NULL);
   createTrackbar( "S_MIN", trackbarWindowName, &S_MIN, 255, NULL);
   createTrackbar( "S_MAX", trackbarWindowName, &S_MAX, 255, NULL);
   createTrackbar( "V_MIN", trackbarWindowName, &V_MIN, 255, NULL);
   createTrackbar( "V_MAX", trackbarWindowName, &V_MAX, 255, NULL);

   cap->left(true);

   Mat image;
   Mat hsvImage;
   Mat thresholdHSVImage;
   Mat contourMask(cap->height(), cap->width(), CV_8UC1, Scalar(0));
   Mat depthMat;
   Mat NaNMask;


   float camera_hfov = 84.14 * (M_PI / 180.0); //measured experimentally
   float camera_vfov = 53.836 * (M_PI / 180.0); //calculated based on hfov and 16/9 aspect ratio
   
   vector< Point2f > target_shape_c; //profile of the target object in mm so we can easily get information about it by using opencv's functions
   Rect targetShapeBound;

   target_shape_c.push_back(Point(0,0));
   target_shape_c.push_back(Point(0,609.6));
   target_shape_c.push_back(Point(50.8,609.6));
   target_shape_c.push_back(Point(50.8,50.8)); //describes vision target shape
   target_shape_c.push_back(Point(762,50.8)); //also this does not work with the data in m
   target_shape_c.push_back(Point(762,609.6));
   target_shape_c.push_back(Point(812.8,609.6));
   target_shape_c.push_back(Point(812.8,0));

   targetShapeBound = boundingRect(target_shape_c);
   Moments targetShapeMoments = moments(target_shape_c, false);
   Point2f targetShapeMC = Point2f( targetShapeMoments.m10/targetShapeMoments.m00 , targetShapeMoments.m01/targetShapeMoments.m00 );

   while(true)
   {
	cap->update();
	cap->getFrame().copyTo(image);

	 imshow ("Normalized Depth", cap->getNormalDepth());
	 imwrite ("image.png", image);

	 cvtColor(image, hsvImage, COLOR_BGR2HSV);
	 generateThreshold(image, thresholdHSVImage,
	       H_MIN, H_MAX, S_MIN, S_MAX, V_MIN, V_MAX);
	 imshow ("HSV threshold", thresholdHSVImage);

	 vector<vector<Point> > recognized_ac;
	 vector<Vec4i> hierarchy;

	 /// Find contours
	 findContours( thresholdHSVImage, recognized_ac, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	 /// Get the mass centers of detected targets
	 Point2f mc;
	 Rect targetRect;
	 for( int i = 0; i < recognized_ac.size(); i++ )
	 {
	    Moments mu = moments( recognized_ac[i], false );
	    mc = Point2f( mu.m10/mu.m00 , mu.m01/mu.m00 ); 
	    //cout << "recognized area (px^2): " << contourArea(recognized_ac[i]) << endl;

	    targetRect = boundingRect(recognized_ac[i]);

	    cap->getDepth().copyTo(depthMat);

	    Mat targetDepthMat(depthMat,targetRect);
	    contourMask.setTo(Scalar(0));
	    drawContours(contourMask,recognized_ac,i,Scalar(255),CV_FILLED); //draw a contour so we can use it as a mask for averaging depth data
	    imshow("Contours mask", contourMask);
	    
	    double depth_z_min = minOfMat(targetDepthMat,contourMask,countPixel);
	    double depth_z_max = minOfMat(targetDepthMat,contourMask,countPixel,true);

	    cout << "Zed depth min: " << depth_z_min << endl;

	    double h_dist_with_min = sqrt( (depth_z_min*depth_z_min) - (goal_height*goal_height) );
	    double h_dist_with_max = sqrt( (depth_z_max*depth_z_max) - ((goal_height+targetRect.height/1000.0)*(goal_height+targetRect.height/1000.0)));

	    //double h_dist_with_avg = sqrt( (depth_z_avg*depth_z_avg) - ((goal_height + targetShapeMC.y/1000.0)*(goal_height + (targetShapeMC.y/1000.0))));

	    double h_dist = (h_dist_with_max + h_dist_with_min) / 2.0;

	    double goal_w_angle = camera_hfov * (targetRect.width / (float)cap->width());
	    double h_angle_to_goal = cos( ( (2*depth_z_min*tan(goal_w_angle)) / (targetShapeBound.width/1000.f) ) );

	    cout << "Horizontal distance to goal: " << h_dist << endl;
	    cout << "Horizontal angle to goal: " << h_angle_to_goal * (180 / M_PI) << endl;
	 }

	 imshow ("BGR", image);
	 imshow ("HSV", hsvImage);
	 waitKey(5);
   }
}
