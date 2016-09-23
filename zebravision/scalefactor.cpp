#include "opencv2_3_shim.hpp"

using namespace std;
using namespace cv;
#if CV_MAJOR_VERSION == 2
using namespace cv::gpu;
#elif CV_MAJOR_VERSION == 3
#include <opencv2/cudawarping.hpp>
using namespace cv::cuda;
#endif

void scalefactor(const Mat &inputimage, const Size &objectsize, const Size &minsize, const Size &maxsize, double scaleFactor, vector<pair<Mat, double> > &scaleInfo)
{
	scaleInfo.clear();
	/*
	Loop multiplying the image size by the scalefactor upto the maxsize	
	Store each image in the images vector
	Store the scale factor in the scales vector 
	*/

	//only works for square image?
	double scale = (double)objectsize.width / minsize.width;

	while(scale > (double)objectsize.width / maxsize.width)
	{	
		//set objectsize.width to scalefactor * objectsize.width
		//set objectsize.height to scalefactor * objectsize.height
		Mat outputimage;
		cv::resize(inputimage, outputimage, Size(), scale, scale);

		// Resize will round / truncate to integer size, recalculate
		// scale using actual results from the resize
		double newscale = max((double)outputimage.rows / inputimage.rows, (double)outputimage.cols / inputimage.cols);
		
		scaleInfo.push_back(make_pair(outputimage, newscale));

		scale /= scaleFactor;		
	}	
}

void scalefactor(const GpuMat &inputimage, const Size &objectsize, const Size &minsize, const Size &maxsize, double scaleFactor, vector<pair<GpuMat, double> > &scaleInfo)
{
	scaleInfo.clear();
	/*
	Loop multiplying the image size by the scalefactor upto the maxsize	
	Store each image in the images vector
	Store the scale factor in the scales vector 
	*/

	//only works for square image?
	double scale = (double)objectsize.width / minsize.width;

	while(scale > (double)objectsize.width / maxsize.width)
	{	
		//set objectsize.width to scalefactor * objectsize.width
		//set objectsize.height to scalefactor * objectsize.height
		GpuMat outputimage;
		cv::cuda::resize(inputimage, outputimage, Size(), scale, scale);

		// Resize will round / truncate to integer size, recalculate
		// scale using actual results from the resize
		double newscale = max((double)outputimage.rows / inputimage.rows, (double)outputimage.cols / inputimage.cols);
		
		scaleInfo.push_back(make_pair(outputimage, newscale));

		scale /= scaleFactor;		
	}	
}

// Create an array of images which are resized from scaleInfoIn by a factor
// of resizeFactor. Used to create a list of d24-sized images from a d12 list. Can't
// use the function above with a different window size since rounding errors will add +/- 1 to
// the size versus just doing 2x the actual size of the d12 calculations
void scalefactor(const Mat &inputimage, const vector<pair<Mat, double> > &scaleInfoIn, int rescaleFactor, vector<pair<Mat, double> > &scaleInfoOut)
{
	scaleInfoOut.clear();
	for (auto it = scaleInfoIn.cbegin(); it != scaleInfoIn.cend(); ++it)
	{
		Mat outputimage;

		Size newSize(it->first.cols * rescaleFactor, it->first.rows * rescaleFactor);
		cv::resize(inputimage, outputimage, newSize);
		// calculate scale from actual size, which will
		// include rounding done to get to integral number
		// of pixels in each dimension
		double scale = max((double)outputimage.rows / inputimage.rows, (double)outputimage.cols / inputimage.cols);
		scaleInfoOut.push_back(make_pair(outputimage, scale));
	}
}

void scalefactor(const GpuMat &inputimage, const vector<pair<GpuMat, double> > &scaleInfoIn, int rescaleFactor, vector<pair<GpuMat, double> > &scaleInfoOut)
{
	scaleInfoOut.clear();
	for (auto it = scaleInfoIn.cbegin(); it != scaleInfoIn.cend(); ++it)
	{
		GpuMat outputimage;

		Size newSize(it->first.cols * rescaleFactor, it->first.rows * rescaleFactor);
		cv::cuda::resize(inputimage, outputimage, newSize);
		// calculate scale from actual size, which will
		// include rounding done to get to integral number
		// of pixels in each dimension
		double scale = max((double)outputimage.rows / inputimage.rows, (double)outputimage.cols / inputimage.cols);
		scaleInfoOut.push_back(make_pair(outputimage, scale));
	}
}

