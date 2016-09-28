#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <sys/time.h>

using namespace std;
using namespace cv;
using namespace cv::cuda;
static double gtod_wrapper(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}


// Be conservative here - if any of the depth values in the target rect
// are in the expected range, consider the rect in range.  Also
// say that it is in range if any of the depth values are negative (i.e. no
// depth info for those pixels)
bool depthInRange(float depth_min, float depth_max, const Mat& detectCheck, bool print = false)
{
    for (int py = 0; py < detectCheck.rows; py++)
    {
        const float *p = detectCheck.ptr<float>(py);
        for (int px = 0; px < detectCheck.cols; px++)
        {
            if (isnan(p[px]) || (p[px] <= 0.0) || ((p[px] <= depth_max) && (p[px] > depth_min)))
            {
				if (print)
					cout << "px = " << px << " py = " << py << endl;
                return true;
            }
        }
    }
	if (print)
		cout << "InRange = false" << endl;
    return false;
}

void printDepthInRange(float depth_min, float depth_max, const Mat& detectCheck)
{
    for (int py = 0; py < detectCheck.rows; py++)
    {
        const float *p = detectCheck.ptr<float>(py);
        for (int px = 0; px < detectCheck.cols; px++)
        {
            if (isnan(p[px]) || (p[px] <= 0.0) || ((p[px] <= depth_max) && (p[px] > depth_min)))
				cout << "1 " << endl;
			else
				cout << "0 " << endl;
        }
		cout << endl;
    }
}

vector<bool> cudaDepthThreshold(const vector<GpuMat> &depthList, const float depthMin, const float depthMax);

int main(void)
{
	RNG rng(12345);

	int count = 0;
	Mat mat(12, 12, CV_32FC1);
	while(1)
	{
		count += 1;
		if ((count % 10000) == 0)
			cout << count << endl;
		double start = gtod_wrapper();
		for (int i = 0; i < 50000; i++)
		{
			rng.fill(mat, RNG::UNIFORM, -100.f, 10000.f);
			const float minRange = rng.uniform(0.1f, 9000.f);
			const float maxRange = rng.uniform(minRange, 9000.f);

			auto cpuResult = depthInRange(minRange, maxRange, mat);
			vector <GpuMat> gpuList;
			gpuList.push_back(GpuMat(mat));
			auto gpuResult = cudaDepthThreshold(gpuList, minRange, maxRange);
			if (cpuResult != gpuResult[0])
			{
				cout << mat <<endl;
				printDepthInRange(minRange, maxRange, mat);
			}

		}
		double end = gtod_wrapper();
		cout << "Time : " << end - start << endl;

#if 0
		if (depthInRange(minRange, maxRange, mat) != depthInRangeGPU(minRange, maxRange, mat))
		{
			cout << minRange << " " << maxRange << endl << mat << endl;
			cout << depthInRange(minRange, maxRange, mat, true) << endl;
			cout << depthInRangeGPU(minRange, maxRange, mat, true) << endl;
			return 0;
		}
#endif
	}
	return 0;
}


