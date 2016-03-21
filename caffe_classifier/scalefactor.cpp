#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;

template<class MatT>
void scalefactor(MatT inputimage, Size objectsize, Size minsize, Size maxsize, double scaleFactor, vector<pair<MatT, double> > &scaleInfo)
{
	scaleInfo.clear();
	/*
	Loop multiplying the image size by the scalefactor upto the maxsize	
	Store each image in the images vector
	Store the scale factor in the scales vector 
	*/

	//only works for square image
	double scale = (double)objectsize.width / minsize.width;

	while(scale > (double)objectsize.width / maxsize.width)
	{	
		//set objectsize.width to scalefactor * objectsize.width
		//set objectsize.height to scalefactor * objectsize.height
		MatT outputimage;
		resize(inputimage, outputimage, Size(), scale, scale);
		
		scaleInfo.push_back(make_pair(outputimage, scale));

		scale /= scaleFactor;		
	}	
}

// Explicitly generate code for Mat and GpuMat options
template void scalefactor(Mat inputimage, Size objectsize, Size minsize, Size maxsize, double scaleFactor, vector<pair<Mat, double> > &scaleInfo); 
template void scalefactor(gpu::GpuMat inputimage, Size objectsize, Size minsize, Size maxsize, double scaleFactor, vector<pair<gpu::GpuMat, double> > &scaleInfo);
