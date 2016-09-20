#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/stream_accessor.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

using std::cout;
using std::endl;

static inline void safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
#define SAFE_CALL(call,msg) safe_cuda_call((call),(msg),__FILE__,__LINE__)

#ifndef cublasSafeCall
#define cublasSafeCall(err) cublasSafeCallWrapper(err, __FILE__, __LINE__)
#endif
void cublasSafeCallWrapper(cublasStatus_t err, const char *file, const int line)
{
	if( CUBLAS_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n", file, line ,err);
		cudaDeviceReset(); assert(0);
	}
}

// Take the output of the ZCA matrix mul - that will
// be a matrix. Each image is a row, each row is the pixels
// in BGRBGRBGR.. order
// Convert that to a flat 1-D array as expected by the neural
// net input stages
__global__ void unflatten_kernel(const cv::gpu::PtrStepSz<float> input,
								 const size_t rows,
								 const size_t cols,
							 	 float *output)
{
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Each image it its own zIndex
	const int zIndex = blockIdx.z * blockDim.z + threadIdx.z;

	//Only valid threads perform memory I/O
	if((xIndex < cols) && (yIndex < rows))
	{
		// yIndex * cols = number of floats per complete
		// filled row
		// add xIndex to get to the correct location in this row
		// Multiply by three to account for R, G, B float values
		//   per col in the input images
		const int flatIdxX = 3*(yIndex * cols + xIndex);
		const float blue  = input(zIndex, flatIdxX + 0);
		const float green = input(zIndex, flatIdxX + 1);
		const float red	  = input(zIndex, flatIdxX + 2);

		// Convert to flat 1-D representation
		// order is [image][color channel][row][col]
		const int chanDist = rows * cols;
		const int idx = zIndex * 3 * chanDist + // 3 channels of row*col pixels per image
			            yIndex * cols +         // select correct row   
						xIndex;                 // and the column in that row

		output[idx]                = blue;      // all the blue comes first
		output[idx +     chanDist] = green;     // then the green 
		output[idx + 2 * chanDist] = red;       // then the red from a given image
	}
}

// Math to add two intermediate steps of mean & stddev 
// See http://www.johndcook.com/blog/skewness_kurtosis/
__device__ void combine_running_totals(float &M1_1, const float M1_2, float &M2_1, const float M2_2, unsigned int &n_1, const unsigned int n_2)
{
	unsigned int combined_n = n_1 + n_2;

	const float delta  = M1_2 - M1_1;
	const float delta2 = delta * delta;

	float combined_M1 = (n_1 * M1_1 + n_2 * M1_2) / combined_n;
	float combined_M2 = M2_1 + M2_2 + delta2 * n_1 * n_2 / combined_n;

	n_1  = combined_n;
	M1_1 = combined_M1;
	M2_1 = combined_M2;
}

// For each input image, calculate the mean and stddev
// of each color channel
// TODO : instead of 16x16 thread block use 12x12 or 24x24 depending
//        on input size. Then combine this with the final step of converting
//        M!/M2 into mean and stddev() at the very end of the kernel 
//        This will eliminate the need for global memory for intermediates 
//        since each image will fit in one block
__global__ void mean_stddev_reduction_kernel(const cv::gpu::PtrStepSz<float> *input,
												   cv::gpu::PtrStepSz<float> output)
{

	// Thread index within block - used for addressing smem below
	const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

	// Shared memory per channel per thread = 1 long, 2 floats.
	// So a 3 channel image needs 3 longs and 6 floats
	// Thread blocks are up to 24x24 images, one thread per pixel
	// TODO : fixme for variable sized thread blocks
	__shared__ float M1[32*32*3];
	__shared__ float M2[32*32*3];
	__shared__ unsigned int n[32*32*3];

	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Each image it its own zIndex
	const int zIndex = blockIdx.z * blockDim.z + threadIdx.z;

	// Only valid threads perform memory I/O
	if((xIndex < input[zIndex].cols) && (yIndex < input[zIndex].rows))
	{
		// xIndex * 3 since col has a blue green and red component
		const float blue  = input[zIndex](yIndex, 3*xIndex);
		const float green = input[zIndex](yIndex, 3*xIndex + 1);
		const float red	  = input[zIndex](yIndex, 3*xIndex + 2);

		// Initialize running average
		M1[tid * 3]     = blue;
		M1[tid * 3 + 1] = green;
		M1[tid * 3 + 2] = red;

		// Initialize pixel count
		n[tid * 3]     = 1;
		n[tid * 3 + 1] = 1;
		n[tid * 3 + 2] = 1;
	}
	else
	{
		// This thread has nothing to contribute
		// to the final result
		n[tid * 3]     = 0;
		n[tid * 3 + 1] = 0;
		n[tid * 3 + 2] = 0;
	}

	M2[tid * 3]     = 0;
	M2[tid * 3 + 1] = 0;
	M2[tid * 3 + 2] = 0;
	
    __syncthreads();

    // do reduction in shared mem
	// For each thread, combine the results from 2 threads
	// down into one. Each pass through the loop eliminates
	// half of the partial results, eventually ending up
	// with just one final result per block
    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
			for (int i = 0; i < 3; i++)
			{
				// Blue, green, red = 3 entries per shared mem array
				const int i1 = 3 * tid + i;
				const int i2 = 3 * (tid + s) + i;
				if (n[i2])
					combine_running_totals(M1[i1], M1[i2], M2[i1], M2[i2], n[i1], n[i2]);
n[i2] = 0;
			}
        }
        __syncthreads();
    }

    // Update M1[0-2] and M2[0-2] with the 
    // mean and stddev of the B, G, R pixels
    if (tid < 3)
	{
		// M1 is the mean already - nothing extra needed
		// calculate stddev from M2 and n
		M2[tid] = sqrt(M2[tid] / n[tid]);
	}
	__syncthreads();

	// Apply global contrast normalization to
	// each input image.
	// For each channel in each image, the mean and stddev has
	// already been calculated
	// For each channel in each pixel, subtract the mean and divide by the stddev
	// Insure only valid threads perform memory I/O
	// If the x/y index for this thread is beyond the
	// number of cols/rows, do nothing
	if((xIndex < input[zIndex].cols) && (yIndex < input[zIndex].rows))
	{
		// xIndex * 3 since col has a blue green and red component
		float blue	= input[zIndex](yIndex, 3 * xIndex);
		float green	= input[zIndex](yIndex, 3 * xIndex + 1);
		float red	= input[zIndex](yIndex, 3 * xIndex + 2);

		blue  = (blue  - M1[0]) / M2[0];
		green = (green - M1[1]) / M2[1];
		red   = (red   - M1[2]) / M2[2];

		// yIndex * input[0].cols = number of floats per complete
		// filled row
		// add xIndex to get to the correct location in this row
		// Multiply by three to account for R, G, B float values
		//   per col in the input images
		const int flatIdxX = 3 * (yIndex * input[zIndex].cols + xIndex);
		output(zIndex, flatIdxX + 0) = blue;
		output(zIndex, flatIdxX + 1) = green;
		output(zIndex, flatIdxX + 2) = red;
	}
}

void cudaZCATransform(const std::vector<cv::gpu::GpuMat> &input, 
		const cv::gpu::GpuMat &weights, 
		cv::gpu::PtrStepSz<float> *dPssIn,
		cv::gpu::GpuMat &dFlattenedImages,
		cv::gpu::GpuMat &zcaOut,
		float *output)
{
	// Create array of PtrStepSz entries corresponding to
	// each GPU mat in input. Copy it to device memory
	cv::gpu::PtrStepSz<float> hPssIn[input.size()];
	for (size_t i = 0; i < input.size(); ++i)
		hPssIn[i] = input[i];
	cudaMemcpy(dPssIn, hPssIn, input.size() * sizeof(cv::gpu::PtrStepSz<float>), cudaMemcpyHostToDevice);

	// Each block is one image
	const dim3 block(32,32);

	// z dimension is number of images
	const dim3 grid(1, 1, input.size());

	// Create a CUDA stream. This lets us queue up a number of
	// cuda calls back to back and then later check to see
	// that they all finished
	cudaStream_t stream;
	SAFE_CALL(cudaStreamCreate(&stream), "ZCA cudaStreamCreate");

	//Launch the first reduction kernel
	// this will output an array of intermediate values
	// in M1 (running average) and M2 (variance * number 
	// of values seen). n is number of values corresponding
	// to each M1 and M2 value.
	mean_stddev_reduction_kernel<<<grid,block,0,stream>>>(dPssIn, dFlattenedImages);
	//SAFE_CALL(cudaStreamSynchronize(stream),"ZCA cudaStreamSynchronize failed");

	// Multiply images by weights to get the ZCA-whitened output
	//gemm(dFlattenedImages, weights, 1.0, cv::gpu::GpuMat(), 0.0, zcaOut, 0, stream);

	zcaOut.create(dFlattenedImages.size(), dFlattenedImages.type());
    cublasHandle_t handle;
    cublasSafeCall( cublasCreate_v2(&handle) );
    cublasSafeCall( cublasSetStream_v2(handle, stream) );

    cublasSafeCall( cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST) );
    const float alphaf = 1.0;
    const float betaf = 0.0;
	cublasSafeCall( cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, weights.cols, dFlattenedImages.rows, weights.rows,
		&alphaf,
		weights.ptr<float>(), static_cast<int>(weights.step / sizeof(float)),
		dFlattenedImages.ptr<float>(), static_cast<int>(dFlattenedImages.step / sizeof(float)),
		&betaf,
		zcaOut.ptr<float>(), static_cast<int>(zcaOut.step / sizeof(float))) );
   cublasSafeCall( cublasDestroy_v2(handle) );

	// Copy to output buffer in the order expected by
	// neural net input
	unflatten_kernel<<<grid,block,0,stream>>>(zcaOut, input[0].rows, input[0].cols, output);

	SAFE_CALL(cudaStreamSynchronize(stream),"ZCA cudaStreamSynchronize failed");
	SAFE_CALL(cudaStreamDestroy(stream), "ZCA cudaStreamDestroy failed");
}
