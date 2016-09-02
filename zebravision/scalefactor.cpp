#include <iostream>
#include <iomanip>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;

template<class MatT>
void scalefactor(const MatT &inputimage, const Size &objectsize, const Size &minsize, const Size &maxsize, double scaleFactor, vector<pair<MatT, double> > &scaleInfo)
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
		MatT outputimage;
		resize(inputimage, outputimage, Size(), scale, scale);

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
template<class MatT>
void scalefactor(const MatT &inputimage, const vector<pair<MatT, double> > &scaleInfoIn, int rescaleFactor, vector<pair<MatT, double> > &scaleInfoOut)
{
	scaleInfoOut.clear();
	for (auto it = scaleInfoIn.cbegin(); it != scaleInfoIn.cend(); ++it)
	{
		MatT outputimage;

		Size newSize(it->first.cols * rescaleFactor, it->first.rows * rescaleFactor);
		resize(inputimage, outputimage, newSize);
		// calculate scale from actual size, which will
		// include rounding done to get to integral number
		// of pixels in each dimension
		double scale = max((double)outputimage.rows / inputimage.rows, (double)outputimage.cols / inputimage.cols);
		scaleInfoOut.push_back(make_pair(outputimage, scale));
	}
}

// Explicitly generate code for Mat and GpuMat options
template void scalefactor(const Mat &inputimage, const Size &objectsize, const Size &minsize, const Size &maxsize, double scaleFactor, vector<pair<Mat, double> > &scaleInfo); 
template void scalefactor(const gpu::GpuMat &inputimage, const Size &objectsize, const Size &minsize, const Size &maxsize, double scaleFactor, vector<pair<gpu::GpuMat, double> > &scaleInfo);

template void scalefactor(const Mat &inputimage, const vector<pair<Mat, double> > &scaleInfoIn, int rescaleFactor, vector<pair<Mat, double> > &scaleInfoOut);
template void scalefactor(const gpu::GpuMat &inputimage, const vector<pair<gpu::GpuMat, double> > &scaleInfoIn, int rescaleFactor, vector<pair<gpu::GpuMat, double> > &scaleInfoOut);
