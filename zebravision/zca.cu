#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/gpu/gpu.hpp>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>

using std::cout;
using std::endl;

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__global__ void global_contrast_normalization_kernel(const cv::gpu::PtrStepSz<float> *input,
									const float *mean,
									const float *stddev,
									cv::gpu::PtrStepSz<float> output)
{
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Each image it its own zIndex
	const int zIndex = blockIdx.z * blockDim.z + threadIdx.z;

	//Only valid threads perform memory I/O
	if((xIndex<input[zIndex].cols) && (yIndex<input[zIndex].rows))
	{
		// xIndex * 3 since col has a blue green and red component
		float blue	= input[zIndex](yIndex, 3 * xIndex);
		float green	= input[zIndex](yIndex, 3 * xIndex + 1);
		float red	= input[zIndex](yIndex, 3 * xIndex + 2);

		blue  = (blue  - mean[3*zIndex + 0])/ stddev[3*zIndex + 0];
		green = (green - mean[3*zIndex + 1])/ stddev[3*zIndex + 1];
		red   = (red   - mean[3*zIndex + 2])/ stddev[3*zIndex + 2];

		// yIndex * input[0].cols = number of floats per complete
		// filled row
		// add xIndex to get to the correct location in this row
		// Multiply by three to account for R, G, B float values
		//   per col in the input images
		const int flatIdxX = 3*(yIndex * input[zIndex].cols + xIndex);
		output(zIndex, flatIdxX + 0) = blue;
		output(zIndex, flatIdxX + 1) = green;
		output(zIndex, flatIdxX + 2) = red;
	}
}


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
	if((xIndex<cols) && (yIndex<rows))
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
			            yIndex * cols +            
						xIndex;

		output[idx]              = blue;
		output[idx +   chanDist] = green;
		output[idx + 2*chanDist] = red;
	}
}

__device__ void combine_running_totals(float &M1_1, float M1_2, float &M2_1, float M2_2, unsigned int &n_1, unsigned int n_2)
{
	unsigned int combined_n = n_1 + n_2;

	const float delta = M1_2 - M1_1;
	const float delta2 = delta * delta;

	float combined_M1 = (n_1 * M1_1 + n_2 * M1_2) / combined_n;
	float combined_M2 = M2_1 + M2_2 + delta2 * n_1 * n_2 / combined_n;

	n_1 = combined_n;
	M1_1 = combined_M1;
	M2_1 = combined_M2;
}

// Shared memory per channel per thread = 1 long, 2 floats.
// So a 3 channel image needs 3 longs and 6 floats
__global__ void mean_stddev_reduction_kernel1(const cv::gpu::PtrStepSz<float> *input,
					float *M1Array,
					float *M2Array,
					unsigned int *nArray)
{
	const unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;

	// Thread index within block - used for addressing smem below
	const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

	// TODO : fixme for variable sized thread blocks
	__shared__ float M1[16*16*3];
	__shared__ float M2[16*16*3];
	__shared__ unsigned int n[16*16*3];

	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Each image it its own zIndex
	const int zIndex = blockIdx.z * blockDim.z + threadIdx.z;

	//Only valid threads perform memory I/O
	if((xIndex<input[zIndex].cols) && (yIndex<input[zIndex].rows))
	{
		// xIndex * 3 since col has a blue green and red component
		const float blue	= input[zIndex](yIndex, 3*xIndex);
		const float green	= input[zIndex](yIndex, 3*xIndex + 1);
		const float red		= input[zIndex](yIndex, 3*xIndex + 2);

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
    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
			for (int i = 0; i < 3; i++)
			{
				// Blue, green, red = 3 entries per shared mem array
				int i1 = 3 * tid + i;
				int i2 = 3 * (tid + s) + i;
				if (n[i2])
					combine_running_totals(M1[i1], M1[i2], M2[i1], M2[i2], n[i1], n[i2]);
			}
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
	{
		for (int i = 0; i < 3; i++)
		{
			M1Array[3 * blockId + i] = M1[i];
			M2Array[3 * blockId + i] = M2[i];
			nArray [3 * blockId + i] = n[i];
		}
	}
}


// Convert each intermediate output into a final output
// No more reductions needed, just some scalar math
__global__ void mean_stddev_reduction_kernel12(const float *M1In, const float *M2In, const unsigned int *nIn,
		float *mean, float *stddev)
{
	// 3 entries B,R,G per thread
	const unsigned int tid = threadIdx.x * 3;
	for (unsigned int i = 0; i < 3; i++)
	{
		mean[tid + i] = M1In[tid + i];
		stddev[tid + i] = sqrt(M2In[tid + i]/nIn[tid + i]);
	}
}


// For 24x24 reductions, there are 4 entries x 3 channels
// left over after a 16x16 thread is reduced (i.e. 24x24 fills
// 4 thread blocks). Reduce those 4 values down to one
__global__ void mean_stddev_reduction_kernel24(const float *M1In, const float *M2In, const unsigned int *nIn,
		float *mean, float *stddev)
{
	float M1[3];
	float M2[3];
	unsigned int n[3];

	// There will be 4 sets of 3 color channel results left to 
	// combine here. Each thread is one image
	const unsigned int tidIn = threadIdx.x * 3 * 4;
	for (unsigned int i = 0; i < 3; i++)
	{
		M1[i] = M1In[tidIn + i];
		M2[i] = M2In[tidIn + i];
		n[i]  = nIn[tidIn + i];
	}

	for (unsigned int j = 1; j < 4; j++)
	{
		for (unsigned int i = 0; i < 3; i++)
		{
			// Blue, green, red = 3 entries per shared mem array
			unsigned idx = tidIn + 3 * j + i;

			combine_running_totals(M1[i], M1In[idx], M2[i], M2In[idx], n[i], nIn[idx]);
		}
	}

	// 3 entries B,R,G per thread
	const unsigned int tidOut = threadIdx.x * 3;
	for (unsigned int i = 0; i < 3; i++)
	{
		mean[tidOut + i] = M1[i];
		stddev[tidOut + i] = sqrt(M2[i]/n[i]);
	}

}

void cudaZCATransform(const std::vector<cv::gpu::GpuMat> &input, 
		const cv::gpu::GpuMat &weights, 
		cv::gpu::PtrStepSz<float> *dPssIn,
		cv::gpu::PtrStepSz<float> *dPssOut,
		cv::gpu::GpuMat &dFlattenedImages,
		cv::gpu::GpuMat &zcaOut,
		cv::gpu::GpuMat &buf,
		float *dMean,
		float *dStddev,
		float *output)
{
	// Create array of PtrStepSz entries corresponding to
	// each GPU mat in input. Copy it to device memory
	cv::gpu::PtrStepSz<float> hPssIn[input.size()];
	for (size_t i = 0; i < input.size(); ++i)
	{
		hPssIn[i] = input[i];
		//cout << "hPssIn[i] r=" << hPssIn[i].rows << " cols=" << hPssIn[i].cols << " step=" << hPssIn[i].step << endl;
	}
	cudaMemcpy(dPssIn, hPssIn, input.size() * sizeof(cv::gpu::PtrStepSz<float>), cudaMemcpyHostToDevice);

	// Specify a reasonable block size
	const dim3 block(16,16);

	// Calculate x & y grid size to cover the whole image
	// z dimension is number of images
	const dim3 grid((input[0].cols + block.x - 1)/block.x, (input[0].rows + block.y - 1)/block.y, input.size());

	// Allocate space for M1, M2, n for each block
	const size_t numBlocks = grid.x * grid.y * grid.z;
	float *d_M1;
	float *d_M2;
	unsigned int *d_n;

	// 3 color channels to keep results for
	// TODO : reduce n down to 1 per block since it is the same
	//        for all 3 channels
	cudaMalloc(&d_M1, 3*numBlocks * sizeof(float));
	cudaMalloc(&d_M2, 3*numBlocks * sizeof(float));
	cudaMalloc(&d_n,  3*numBlocks * sizeof(unsigned int ));

	//Launch the first reduction kernel
	// this will output an array of intermediate values
	// in M1 (running average) and M2 (variance * number 
	// of values seen). n is number of values corresponding
	// to each M1 and M2 value.
	mean_stddev_reduction_kernel1<<<grid,block>>>(dPssIn, d_M1, d_M2, d_n);

	//Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(),"Stddev/mean reduction kernel launch failed");

	// Second reduction generates mean and stddev values
	// 12x12 fit in a single block so there's no
	// actual reduction to do, just convert M1/M2 to 
	// mean and stddev.
	// 24x24 doesn't fit in a block so this reduction takes
	// the intermediate results from each block and combines
	// them into the final values
	if (numBlocks == input.size())
		mean_stddev_reduction_kernel12<<<1, input.size()>>>(d_M1, d_M2, d_n, dMean, dStddev);
	else
		mean_stddev_reduction_kernel24<<<1, input.size()>>>(d_M1, d_M2, d_n, dMean, dStddev);
	SAFE_CALL(cudaDeviceSynchronize(),"Stddev/mean final kernel Launch failed");

	// Generate a mat with space for all of the images 
	// in this batch.  Each row is a flattened version
	// of the image - rows are concatenated together
	// to turn the image into a 1-D array
	global_contrast_normalization_kernel<<<grid,block>>>(dPssIn, dMean, dStddev, dFlattenedImages);
	SAFE_CALL(cudaDeviceSynchronize(),"GCN kernel launch failed");

	gemm(dFlattenedImages, weights, 1.0, buf, 0.0, zcaOut);

#if 0
	output.clear();

	// Turn each row back into a 2-d mat with 3 float color channels
	// Create array of PtrStepSz entries in device memory
	cv::gpu::PtrStepSz<float> hPssOut[input.size()];
	for (size_t i = 0; i < input.size(); ++i)
	{
		output.push_back(cv::gpu::GpuMat(input[0].rows, input[0].cols, CV_32FC3));
		hPssOut[i] = output[i];
		//cout << "hPssOut[i] r=" << hPssOut[i].rows << " cols=" << hPssOut[i].cols << " step=" << hPssOut[i].step << endl;
	}
	cudaMemcpy(dPssOut, hPssOut, input.size() * sizeof(cv::gpu::PtrStepSz<float>), cudaMemcpyHostToDevice);
#endif
	unflatten_kernel<<<grid,block>>>(zcaOut, input[0].rows, input[0].cols, output);
	SAFE_CALL(cudaDeviceSynchronize(),"GCN kernel launch failed");

	SAFE_CALL(cudaFree(d_M1),"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_M2),"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_n),"CUDA Free Failed");

}

