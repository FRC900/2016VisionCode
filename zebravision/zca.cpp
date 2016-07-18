// Class for performing zero-phase component analysis & transforms. 
// The goal is to transform input such that the data is zero-mean,
// that individual features are uncorrelated and all have the 
// same variance. This process is known as whitening or sphering. 
// Whitening takes cues from so-called white noise, where the
// next value in a series can't be predicted from previous 
// sample. Sphering refers to the appearance of graphs of the 
// processed data, which look like spherical blobs centered around
// the origin.
// ZCA is a specific type of principal component analysis which 
// performs this transformation with the minimum amount of change
// to the original data. Thus ZCA-whitened images resemble 
// the original input in some sense - this isn't true for the
// general case of possible PCA whitening.
// PCA takes a set of data and transforms it so that the
// variables are uncorrelated.  This transformation takes an N
// dimensional set of data and rotates it to a new coordinate system
// The first axis of the rotated data will be chosen such that
// it accounts for the largest variation in the data. 
// Subsquent dimensions are ones which account for less and 
// less variation.  Each axis is called a principal component,
// and has both a unit vector (where that axis points) and a separate
// magnitude showing how much that particular component impacts the
// overall data value.
// Imagine a 2d set of data which is essentaily y = x +/- a small noise
// component. That is, the data is centered around y=x with small variations
// above and below.  PCA of that data set would rotate it so the
// new primary axis would be the line y=x. The second axis would
// be orthoganl to that and would quantify how much noise each
// data point had - how far it was displaced from the new primary
// y=x axis.  
// Once PCA is applied to a data set, the data points are decorrelated
// with each other.  To achieve the second goal of uniform variance
// the magnitude of each principal component is rescaled by the 
// appropriate amount.
// In this application, the each color channel of each pixel is
// a dimension. So a 12x12 3-channel color image would have 12x12x3 =
// 432 dimensions. There are a number of correlations in natural
// images which whitening can eliminate - colors of nearby pixels
// tend to be similar, the channels of each pixel can be correlated
// due to lighting, and so on.  Removing those correlations which appear
// in every image lets neural net classifier focus on correlations
// that are indicitive of objects we're actually interested in.
//
// Additional references :
// http://ufldl.stanford.edu/wiki/index.php/PCA and following pages on that wiki
// http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/
// http://eric-yuan.me/ufldl-exercise-pca-image/
// http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
// Additional references :

#include <string>
#include "zca.hpp"

using namespace std;
using namespace cv;

// Using the input images provided, generate a ZCA transform
// matrix.  Images will be resized to the requested size 
// before processing (the math doesn't work if input images
// are variable sized).  epsilon is a small positive value added
// to the magnitude of principal components before taking the
// square root of them - since many of these values are quite
// small the square root of them becomes numerically instable.
ZCA::ZCA(const vector<Mat> &images, const Size &size, float epsilon) :
	size_(size),
	epsilon_(epsilon)
{
	if (images.size() == 0)
		return;

	Mat workingMat;
	// For each input image, resize to constant size
	// convert to a floating point mat
	// flatten to a single channel, 1 row matrix
	// Normalize data to (0,1) range
	// Push that new row onto tmpMat
	Mat tmpImg;
	for (auto it = images.cbegin(); it != images.cend(); ++it)
	{
		resize(*it, tmpImg, size_);
		tmpImg.convertTo(tmpImg, CV_32FC3, 1.0/255.0);
		workingMat.push_back(tmpImg.reshape(1, 1));
	}
	tmpImg.release();
	// Transpose so each image is its own column 
	// rather than its own row 
	workingMat = workingMat.t();

	// Find the mean value of each column (i.e. each input image)
	Mat colMean;
	reduce(workingMat, colMean, 0, CV_REDUCE_AVG);

	// Subtract mean value for a given image from
	// each of the pixels in that image to
	// make the data from each image 0-mean
	for (int i = 0; i < workingMat.cols; i++)
		subtract(workingMat.col(i), colMean.at<float>(i), workingMat.col(i));
	colMean.release();

	// sigma is the covariance matrix of the 
	// input data
	Mat sigma = (workingMat * workingMat.t()) / (float)(workingMat.cols - 1.);
	workingMat.release();

	SVD svd;
	Mat svdW; // eigenValues - magnitude of each principal component
	Mat svdU; // eigenVectors - where each pricipal component points
	Mat svdVT;
	svd.compute(sigma, svdW, svdU, svdVT, SVD::FULL_UV);
	
	// Add small epsilon to prevent sqrt(small number)
	// numerical instability
	// Take square root of each element, convert
	// from vector into diagonal array
	svdW += epsilon;
	sqrt(svdW, svdW);
	svdW = 1.0 / svdW;
	Mat svdS = Mat::diag(svdW);

	// Weights are U * S * U'
	weights_ = svdU * svdS * svdU.t();
	weightsGPU_.upload(weights_);
}

// Transform a typical 8 bit image as read from file
// Return the same 8UC3 type
Mat ZCA::Transform8UC3(const Mat &input)
{
	Mat ret;
	Mat tmp;
	input.convertTo(tmp, CV_32FC3, 1/255.0);
	// Convert back to uchar array with correct 0 - 255 range
	// This turns it into a "normal" image file which
	// can be processed and visualized using typical
	// tools
	Transform32FC3(tmp).convertTo(ret, CV_8UC3, 255.0, 127.0);

	return ret;
}

// Transform a typical 8 bit image as read from file
// Return the same 8UC3 type
vector<Mat> ZCA::Transform8UC3(const vector<Mat> &input)
{
	vector<Mat> f32List;
	Mat tmp;

	// Create an intermediate vector of f32 versions
	// of the input image. Scale them so the values
	// are between 0 and 1 as the 32FC3 transform expects
	for (auto it = input.cbegin(); it != input.cend(); ++it)
	{
		it->convertTo(tmp, CV_32FC3, 1/255.0);
		f32List.push_back(tmp);
	}

	// Do the transform 
	vector <Mat> f32Ret = Transform32FC3(f32List);

	// Convert back to uchar array with correct 0 - 255 range
	// This turns it into a "normal" image file which
	// can be processed and visualized using typical
	// tools
	vector <Mat> ret;
	for (auto it = input.cbegin(); it != input.cend(); ++it)
	{
		it->convertTo(tmp, CV_8UC3, 255.0, 127.0);
		ret.push_back(tmp);
	}

	return ret;
}

// Expects a 32FC3 mat as input
// Pixels should be scaled between 0.0 and 1.0
Mat ZCA::Transform32FC3(const Mat &input)
{
	// Resize input
	Mat output;
	Mat final;
	if (input.size() != size_)
		resize(input, output, size_);
	else 
		output = input;

	// Convert to flat 1-dimensional 1-channel vector
	output = output.reshape(1, size_.area() * output.channels());
		
	// Convert to 0-mean
	output -= cv::mean(output)[0];

	// Apply ZCA transform matrix
	if (!weightsGPU_.empty())
	{
		gm_.upload(output);
		cv::gpu::gemm(weightsGPU_, gm_, 1.0, buf_, 0.0, gmOut_);
		gmOut_.download(final);
	}
	else if (!weights_.empty())
		gemm(weights_, output, 1.0, Mat(), 0.0, final);

	// Turn back into a 2-d mat with 3 float color channels
	// Range is same as input : 0.0 to 1.0
	return final.reshape(input.channels(), size_.height);
}

vector<Mat> ZCA::Transform32FC3(const vector<Mat> &input)
{
	Mat output;
	Mat work;
	for (auto it = input.cbegin(); it != input.cend(); ++it)
	{
		if (it->size() != size_)
			resize(*it, output, size_);
		else 
			// need clone so mat is contiguous - 
			// reshape won't work otherwise
			output = it->clone(); 
		
		work.push_back(output.reshape(1, 1));
	}
	work=work.t();
	// Find the mean value of each column (i.e. each input image)
	Mat colMean;
	reduce(work, colMean, 0, CV_REDUCE_AVG);

	// Subtract mean value for a given image from
	// each of the pixels in that image to
	// make the data from each image 0-mean
	for (int i = 0; i < work.cols; i++)
		subtract(work.col(i), colMean.at<float>(i), work.col(i));
	colMean.release();

	// Apply ZCA transform matrix
	if (!weightsGPU_.empty())
	{
		gm_.upload(work);
		gpu::gemm(weightsGPU_, gm_, 1.0, buf_, 0.0, gmOut_);

		gmOut_.download(output);
	}
	else if (!weights_.empty())
		gemm(weights_, work, 1.0, Mat(), 0.0, output);

	output = output.t();

	vector<Mat> ret;
	// Each row is a different input image
	for (int i = 0; i < output.rows; i++)
		// Turn each row back into a 2-d mat with 3 float color channels
		// Range is same as input : 0.0 to 1.0
		ret.push_back(output.row(i).reshape(input[i].channels(), size_.height));

	return ret;
}

// Load a previously calcuated set of weights from file
ZCA::ZCA(const char *xmlFilename)
{
	try 
	{
		FileStorage fs(xmlFilename, FileStorage::READ);
		if (fs.isOpened())
		{
			fs["ZCASize"] >> size_;
			fs["ZCAWeights"] >> weights_;
			if (!weights_.empty() && (gpu::getCudaEnabledDeviceCount() > 0))
				weightsGPU_.upload(weights_);
			fs["ZCAEpsilon"] >> epsilon_;
		}
		fs.release();
	}
	catch (const std::exception &e)
	{
		return;
	}
}

// Save calculated weights to a file
void ZCA::Write(const char *xmlFilename) const
{
	FileStorage fs(xmlFilename, FileStorage::WRITE);
	fs << "ZCASize" << size_;
	fs << "ZCAWeights" << weights_;
	fs << "ZCAEpsilon" << epsilon_;
	fs.release();
}
