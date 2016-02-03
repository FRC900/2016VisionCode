#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

template<class MatT>
void scalefactor(MatT inputimage, cv::Size objectsize, cv::Size minsize, cv::Size maxsize, double scaleFactor, std::vector<std::pair<MatT, double> > &scaleInfo)
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
		resize(inputimage, outputimage, cv::Size(), scale, scale);
		
		scaleInfo.push_back(std::make_pair(outputimage, scale));

		scale /= scaleFactor;		
	
	}	
}

// Explicitly generate code for Mat and GpuMat options
template void scalefactor(cv::Mat inputimage, cv::Size objectsize, cv::Size minsize, cv::Size maxsize, double scaleFactor, std::vector<std::pair<cv::Mat, double> > &scaleInfo); 
template void scalefactor(cv::gpu::GpuMat inputimage, cv::Size objectsize, cv::Size minsize, cv::Size maxsize, double scaleFactor, std::vector<std::pair<cv::gpu::GpuMat, double> > &scaleInfo);
