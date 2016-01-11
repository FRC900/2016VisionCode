#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

template<class MatT>
void scalefactor(MatT inputimage, cv::Size objectsize, cv::Size minsize, cv::Size maxsize, float scaleFactor, std::vector<std::pair<MatT, float> > &scaleInfo)
{
	scaleInfo.clear();
	/*
	Loop multiplying the image size by the scalefactor upto the maxsize	
	Store each image in the images vector
	Store the scale factor in the scales vector 
	*/

	//for(Size i = objectsize; i < maxsize;)
	
	//only works for square image
	float scale = (float)objectsize.width / minsize.width;

	while(scale > (float)objectsize.width / maxsize.width)
	{	
		//set objectsize.width to scalefactor * objectsize.width
		//set objectsize.height to scalefactor * objectsize.height
		MatT outputimage;
		resize(inputimage, outputimage, cv::Size(), scale, scale);
		
		scaleInfo.push_back(std::pair<MatT, float>(outputimage, scale));

		scale /= scaleFactor;		
	
	}	
}

// Explicitly generate code for Mat and GpuMat options
template void scalefactor(cv::Mat inputimage, cv::Size objectsize, cv::Size minsize, cv::Size maxsize, float scaleFactor, std::vector<std::pair<cv::Mat, float> > &scaleInfo); 
template void scalefactor(cv::gpu::GpuMat inputimage, cv::Size objectsize, cv::Size minsize, cv::Size maxsize, float scaleFactor, std::vector<std::pair<cv::gpu::GpuMat, float> > &scaleInfo);
