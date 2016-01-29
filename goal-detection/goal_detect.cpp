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

int H_MIN = 60; //60-95 is a good range for bright green
int H_MAX = 95;
int S_MIN =  180;
int S_MAX = 255;
int V_MIN =  67;
int V_MAX = 255;
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
   Mat contours_black(cap->height(), cap->width(), CV_8UC1, Scalar(0,0,0));
   while(true)
   {
	cap->update();
	cap->getFrame().copyTo(image);

	imshow ("BGR", image);

	 cvtColor(image, hsvImage, COLOR_BGR2HSV);
	 generateThreshold(image, thresholdHSVImage,
	       H_MIN, H_MAX, S_MIN, S_MAX, V_MIN, V_MAX);
	 imshow ("HSV threshold", thresholdHSVImage);

	 vector<vector<Point> > contours;
	 vector<Vec4i> hierarchy;

	 /// Find contours
	 findContours( thresholdHSVImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	 /// Get the mass centers of detected targets
	 vector<Point2f> mc( contours.size() );
	 for( int i = 0; i < contours.size(); i++ )
	 { 
	    Moments mu = moments( contours[i], false );
	    mc[i] = Point2f( mu.m10/mu.m00 , mu.m01/mu.m00 ); 
	 }

	 /// Draw contours
	 double maxMC = 0.0;
	 double minMC = DBL_MAX; 
	 for( int i = 0; i< contours.size(); i++ ) //runs for each contour that it found
	 {
	       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	       drawContours( contours_black, contours, i, color, 2, 8, hierarchy, 0, Point() );
	       circle( contours_black, mc[i], 4, color, -1, 8, 0 );
	       
	  	cout << "Countour " << i << " moments: " << mc[i].x << " , " << mc[i].y << endl;
	 }
	 
	 imshow ("HSV", hsvImage);
	 imshow ("Contours", contours_black);
	 waitKey(5);
   }
}
