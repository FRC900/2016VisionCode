#include <vector>
#include "cuda_utils.hpp"
#include "opencv2_3_shim.hpp"

#if CV_MAJOR_VERSION == 2
using cv::gpu::PtrStepSz;
#elif CV_MAJOR_VERSION == 3
using cv::cuda::PtrStepSz;
#endif

// Given a depth map in input, see if any value is
// in the range between depthMin and depthMax.  If
// so, set answer to true. If all pixels fall outside
// the range, set answer to false.
__global__ void depth_threshold_kernel(const PtrStepSz<float> *input,
									   const float depthMin,
									   const float depthMax,
									   bool *answer)
{
	// Thread index within block - used for addressing smem below
	const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ bool inRange[16*16];

	// 2D pixel index of current thread
	const int xIndex = threadIdx.x;
	const int yIndex = threadIdx.y;

	const int imgIndex = blockIdx.x;

	// Only valid threads perform memory I/O
	if((xIndex < input[0].cols) && (yIndex < input[0].rows))
	{
		// Be conservative here - if any of the depth values in the 
		// target rect are in the expected range, consider the entire 
		// rect in range.  Also say that it is in range if any of the 
		// depth values are negative (i.e. no depth info for those pixels)
		const float depth = input[imgIndex](yIndex, xIndex);
		if (isnan(depth) || (depth <= 0.0) || ((depth > depthMin) && (depth < depthMax)))
			inRange[tid] = true;
		else
			inRange[tid] = false;
	}
	else
	{
		// Set values outside the range of the image
		// to false. This will make them ignored in
		// the reduction down to a single compare value
		inRange[tid] = false;
	}

	// Let all threads finish the compare and put
	// their results in shared mem
    __syncthreads();

    // do reduction in shared mem
	// For each thread, combine the results from 2 threads
	// down into one. Each pass through the loop eliminates
	// half of the partial results, eventually ending up
	// with just one final result per block
    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1)
    {
		if (inRange[0])
			break;

		// Basically just propagate any true values
		// down to thread 0 - only return false
		// if the entire set of compares was false
        if ((tid < s) && inRange[tid + s])
			inRange[tid] = true;
        __syncthreads();
    }

    if (tid == 0)
		answer[imgIndex] = inRange[0];
}

__host__ std::vector<bool> cudaDepthThreshold(const std::vector<GpuMat> &depthList, const float depthMin, const float depthMax)
{
	// Create array of PtrStepSz entries corresponding to
	// each GPU mat in depthList. Copy it to device memory

	PtrStepSz<float> hPssIn[depthList.size()];
	for (size_t i = 0; i < depthList.size(); ++i)
		hPssIn[i] = depthList[i];
	PtrStepSz<float> *dPssIn;
	cudaSafeCall(cudaMalloc(&dPssIn, depthList.size() * sizeof(*dPssIn)), "cudaMalloc threshold dPssIn");
	cudaSafeCall(cudaMemcpy(dPssIn, hPssIn, depthList.size() * sizeof(PtrStepSz<float>), cudaMemcpyHostToDevice), "cudaMemcpy dPssIn");
	
	bool *dResult;
	cudaSafeCall(cudaMalloc(&dResult, depthList.size() * sizeof(bool)), "cudaMalloc threshold result");

	// Each block is one depth
	// Set the block size to the smallest power
	// of two large enough to hold an depth
	const dim3 block(16, 16);

	// each block is 1 image
	const dim3 grid(depthList.size());

	depth_threshold_kernel<<<grid, block>>>(dPssIn, depthMin, depthMax, dResult);
	cudaSafeCall(cudaDeviceSynchronize(), "depthThreshold cudaDeviceSynchronize failed");

	bool hResult[depthList.size()];
	cudaSafeCall(cudaMemcpy(&hResult, dResult, sizeof(bool) * depthList.size(), cudaMemcpyDeviceToHost), "cudaMemcpy depth result");
	cudaSafeCall(cudaFree(dPssIn), "depthThreshold cudaFree");
	cudaSafeCall(cudaFree(dResult), "depthThreshold cudaFree");

	return std::vector<bool>(hResult, hResult + depthList.size());
}
