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
#include <cuda_runtime.h>
#include "zca.hpp"

//#define DEBUG_TIME
#ifdef DEBUG_TIME
#include <sys/time.h>
double gtod_wrapper(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#endif

using namespace std;
using namespace cv;
using namespace cv::gpu;

// Using the input images provided, generate a ZCA transform
// matrix.  Images will be resized to the requested size 
// before processing (the math doesn't work if input images
// are variable sized).  epsilon is a small positive value added
// to the magnitude of principal components before taking the
// square root of them - since many of these values are quite
// small the square root of them becomes numerically unstable.
ZCA::ZCA(const vector<Mat> &images, const Size &size, 
		 float epsilon, bool globalContrastNorm) :
	size_(size),
	epsilon_(epsilon),
	overallMin_(numeric_limits<double>::max()),
	overallMax_(numeric_limits<double>::min()),
	globalContrastNorm_(globalContrastNorm)
{
	if (images.size() == 0)
		return;

	// Build the transposed mat since that's the more
	// natural way opencv works - it is easier to add
	// new images as a new row but the math requires
	// them to columns. Since the later steps require 
	// both the normal and transposed versions of the
	// data, generate the transpose and then "untranspose"
	// it to get the correct one
	Mat workingMatT;

	// For each input image, convert to a floating point mat
	//  and resize to constant size
	// Find the mean and stddev of each color channel. Subtract
	// out the mean and divide by stddev to get to 0-mean
	// 1-stddev for each channel of the image - this 
	// helps normalize contrast between
	// images in different lighting conditions 
	// flatten to a single channel, 1 row matrix
	Mat resizeImg;
	Mat tmpImg;
	Scalar mean;
	Scalar stddev;
	for (auto it = images.cbegin(); it != images.cend(); ++it)
	{
		it->convertTo(resizeImg, CV_32FC3);
		resize(resizeImg, tmpImg, size_);
		meanStdDev(tmpImg, mean, stddev);
		subtract(tmpImg, mean, tmpImg);
		if (globalContrastNorm_)
			divide(tmpImg, stddev, tmpImg);
		else
			divide(tmpImg, Scalar(255., 255., 255.), tmpImg); // TODO : remove this, or use uniform scaling for everything?
		tmpImg = tmpImg.reshape(1, 1);
		workingMatT.push_back(tmpImg.clone());
	}
	resizeImg.release();
	tmpImg.release();

	// Transpose so each image is its own column 
	// rather than its own row 
	Mat workingMat = workingMatT.t();

	// sigma is the covariance matrix of the 
	// input data
	// Literature disagrees on dividing by cols or cols-1
	// Since we're using a large number of input
	// images it really doesn't matter that much
	Mat sigma = (workingMat * workingMatT) / (float)workingMat.cols;

	workingMatT.release();

	SVD svd;
	Mat svdW; // eigenValues - magnitude of each principal component
	Mat svdU; // eigenVectors - where each pricipal component points
	Mat svdVT;
	svd.compute(sigma, svdW, svdU, svdVT, SVD::FULL_UV);
	
	//cout << "svdW" << endl << svdW << endl;
	// Add small epsilon to prevent sqrt(small number)
	// numerical instability. Larger epsilons have a
	// bigger smoothing effect
	// Take square root of each element, convert
	// from vector into diagonal array
	svdW += epsilon;
	sqrt(svdW, svdW);
	svdW = 1.0 / svdW;
	Mat svdS = Mat::diag(svdW);

	// Weights are U * S * U'
	weights_ = svdU * svdS * svdU.t();
	weightsGPU_.upload(weights_);

	// Transform the input images. Grab
	// a range of pixel values and use this
	// to convert back from floating point to
	// something in the range of 0-255
	// Don't want to use the full range of the
	// pixels since outliers will squash the range
	// most pixels end up in to just a few numbers.
	// Instead use the mean +/- 2.25 std deviations
	// This should allow full range representation of
	// > 96% of the pixels
	Mat transformedImgs = weights_ * workingMat;
	meanStdDev(transformedImgs, mean, stddev);
	cout << "transformedImgs mean/stddev " << mean(0) << " " << stddev(0) << endl;
	overallMax_ = mean(0) + 2.25*stddev(0);
	overallMin_ = mean(0) - 2.25*stddev(0);

	// Formula to convert is uchar_val = alpha * float_val + beta
	// This will convert the majority of floating
	// point values into a 0-255 range that fits
	// into a normal 8UC3 mat without saturating
	cout << "Alpha / beta " << alpha() << " "<< beta() << endl;
}

// Transform a typical 8 bit image as read from file
// Return the same 8UC3 type
// Just a wrapper around the faster version
// which does a batch at a time. Why aren't
// you using that one instead?
Mat ZCA::Transform8UC3(const Mat &input)
{
	vector<Mat> inputs;
	inputs.push_back(input);
	vector<Mat> outputs = Transform8UC3(inputs);
	return outputs[0].clone();
}

// Transform a typical 8 bit images as read from files
// Return the same 8UC3 type
vector<Mat> ZCA::Transform8UC3(const vector<Mat> &input)
{
	vector<Mat> f32List;
	Mat tmp;

	// Create an intermediate vector of f32 versions
	// of the input image. 
	for (auto it = input.cbegin(); it != input.cend(); ++it)
	{
		it->convertTo(tmp, CV_32FC3);
		f32List.push_back(tmp.clone());
	}

	// Do the transform 
	vector <Mat> f32Ret = Transform32FC3(f32List);

	// Convert back to uchar array with correct 0 - 255 range
	// This turns it into a "normal" image file which
	// can be processed and visualized using typical
	// tools
	// The float version will have values in a range which
	// can't be exactly represented by the uchar version 
	// (e.g. negative numbers, numbers larger than 255, etc).
	// Scaling by alpha/beta will shift the range of
	// float values to 0-255 (techinally, it'll move ~96%
	// of the value into that range, the rest will be saturated
	// to 0 or 255).  When training using these ucar images,
	// be sure to undo the scaling so they are converted back
	// to the correct float values
	vector <Mat> ret;
	for (auto it = f32Ret.cbegin(); it != f32Ret.cend(); ++it)
	{
		it->convertTo(tmp, CV_8UC3, alpha(), beta());
		ret.push_back(tmp.clone());
	}

	return ret;
}

// Expects a 32FC3 mat as input
// Just a wrapper around the faster version
// which does a batch at a time. Why aren't
// you using that one instead?
Mat ZCA::Transform32FC3(const Mat &input)
{
	vector<Mat> inputs;
	inputs.push_back(input);
	vector<Mat> outputs = Transform32FC3(inputs);
	return outputs[0].clone();
}

// Transform a vector of input images in floating
// point format using the weights loaded
// when this object was initialized
vector<Mat> ZCA::Transform32FC3(const vector<Mat> &input)
{
#ifdef DEBUG_TIME
	double start = gtod_wrapper();
#endif
	Mat output;
	Mat work;
	// Create a large mat holding all of the pixels
	// from all of the input images.
	// Each row is data from one image. Each image
	// is flattened to 1 channel of interlaved B,G,R
	// values.  
	// Global contrast normalization is applied to
	// each image - subtract the mean and divide
	// by the standard deviation separately for each
	// channel. That way each image is normalized to 0-mean 
	// and a standard deviation of 1 before running it
	// through ZCA weights.
	for (auto it = input.cbegin(); it != input.cend(); ++it)
	{
		if (it->size() != size_)
			resize(*it, output, size_);
		else 
			// need clone so mat is contiguous - 
			// reshape won't work otherwise
			output = it->clone();

		Scalar mean;
		Scalar stddev;
		meanStdDev(output, mean, stddev);

		// If GCN is disabled, just scale the values into
		// a range from 0-1.  
		if (!globalContrastNorm_)
			stddev = Scalar(255., 255., 255., 255.);

		for (int r = 0; r < output.rows; r++)
		{
			Vec3f *p = output.ptr<Vec3f>(r);
			for (int c = 0; c < output.cols; c++)
			{
				p[c][0] = (p[c][0] - mean[0]) / stddev[0];
				p[c][1] = (p[c][1] - mean[1]) / stddev[1];
				p[c][2] = (p[c][2] - mean[2]) / stddev[2];
			}
		}

		// Reshape flattens the image to 1 channel, 1 row.
		// Push that row into the bottom of work
		work.push_back(output.reshape(1,1));
	}
#ifdef DEBUG_TIME
	double end = gtod_wrapper();
	cout << "create work " << end - start << endl;
	start = gtod_wrapper();
#endif

	// Apply ZCA transform matrix
	// Math here is weights * images = output images
	// This works if each image is a column of data
	// The natural way to add data above using push_back
	//  creates a transpose of that instead (i.e. each image is its
	//  own row rather than its own column).  Take advantage
	//  of the identiy (AB)^T = B^T A^T.  A=weights, B=images
	// Since we want to pull images apart in the same transposed
	// order, this saves a few transposes and gives a
	// slight performance bump.

	// GPU is faster so use it if it exists.
	if (!weightsGPU_.empty())
	{
		gm_.upload(work);
		gpu::gemm(gm_, weightsGPU_, 1.0, buf_, 0.0, gmOut_);

		gmOut_.download(output);
	}
	else if (!weights_.empty())
		gemm(work, weights_, 1.0, Mat(), 0.0, output);

#ifdef DEBUG_TIME
	end = gtod_wrapper();
	cout << "gemm " << end - start << endl;
	start = gtod_wrapper();
#endif

	// Matrix comes out transposed - instead
	// of an image per column it is an image per row.
	// That's a natural fit for taking them apart
	// back into images, though, so it save some time
	// not having to transpose the output

	// Each row is a different input image,
	// put them each into their own Mat
	vector<Mat> ret;
	for (int i = 0; i < output.rows; i++)
	{
		// Turn each row back into a 2-d mat with 3 float color channels
		ret.push_back(output.row(i).reshape(input[i].channels(), size_.height));
	}

#ifdef DEBUG_TIME
	end = gtod_wrapper();
	cout << "Create ret" << end - start << endl;
#endif
	return ret;
}

void cudaZCATransform(const std::vector<cv::gpu::GpuMat> &input, 
		const cv::gpu::GpuMat &weights, 
		cv::gpu::PtrStepSz<float> *dPssIn,
		cv::gpu::GpuMat &gm,
		cv::gpu::GpuMat &gmOut,
		cv::gpu::GpuMat &buf,
		float *dMean,
		float *dStddev,
		float *output);

// Transform a vector of input images in floating
// point format using the weights loaded
// when this object was initialized
void ZCA::Transform32FC3(const vector<GpuMat> &input, float *dest)
{
	vector<GpuMat> foo;
	for (auto it = input.cbegin(); it != input.cend(); ++it)
	{
		if (it->size() != size_)
		{
			foo.push_back(GpuMat());
			resize(*it, foo[foo.size() - 1], size_);
		}
		else 
		{
			foo.push_back(*it);
		}
	}
	cudaZCATransform(foo, weightsGPU_, dPssIn_, gm_, gmOut_, buf_, dMean_, dStddev_, dest);
}

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)


// Load a previously calcuated set of weights from file
ZCA::ZCA(const char *xmlFilename, size_t batchSize) :
	dPssIn_(NULL),
	dMean_(NULL),
	dStddev_(NULL)
{
	try 
	{
		FileStorage fs(xmlFilename, FileStorage::READ);
		if (fs.isOpened())
		{
			fs["ZCASize"] >> size_;
			fs["ZCAWeights"] >> weights_;
	
			// Transpose these once here to save doing
			// it every time in the calcuation step
			weights_ = weights_.t();
			if (!weights_.empty() && (gpu::getCudaEnabledDeviceCount() > 0))
				weightsGPU_.upload(weights_);

			fs["ZCAEpsilon"] >> epsilon_;
			fs["OverallMin"] >> overallMin_;
			fs["OverallMax"] >> overallMax_;
			fs["GlobalContrastNorm"] >> globalContrastNorm_;
		}
		fs.release();
	}
	catch (const std::exception &e)
	{
		return;
	}

	if (!weightsGPU_.empty())
	{
		SAFE_CALL(cudaMalloc(&dPssIn_, batchSize * sizeof(cv::gpu::PtrStepSz<float>)), "cudaMalloc dPssIn");
		gm_ = GpuMat(batchSize, size_.area() * 3, CV_32FC1);
		SAFE_CALL(cudaMalloc(&dMean_,   3 * batchSize * sizeof(float)), "cudaMalloc mean");
		SAFE_CALL(cudaMalloc(&dStddev_, 3 * batchSize * sizeof(float)), "cudaMalloc stddev");
	}
}

ZCA::~ZCA()
{
	if (dPssIn_)
		SAFE_CALL(cudaFree(dPssIn_), "cudaFree dPssIn");
	if (dMean_)
		SAFE_CALL(cudaFree(dMean_), "cudaFree dMean");
	if (dStddev_)
		SAFE_CALL(cudaFree(dStddev_), "cudaFree dStddev");
}

// Save calculated weights to a file
void ZCA::Write(const char *xmlFilename) const
{
	FileStorage fs(xmlFilename, FileStorage::WRITE);
	fs << "ZCASize" << size_;
	fs << "ZCAWeights" << weights_;
	fs << "ZCAEpsilon" << epsilon_;
	fs << "OverallMin" << overallMin_;
	fs << "OverallMax" << overallMax_;
	fs << "GlobalContrastNorm" << globalContrastNorm_;
	fs.release();
}

// Generate constants to convert from float
// mat back to 8UC3 one.  
double ZCA::alpha(int maxPixelValue) const
{
	double range = overallMax_ - overallMin_;

	return maxPixelValue / range;
}
double ZCA::beta(void) const
{
	return -overallMin_ * alpha();
}

Size ZCA::size(void) const
{
	return size_;
}
