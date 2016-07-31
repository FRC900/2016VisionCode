#include <string>
#include "zca.hpp"

#include "random_subimage.hpp"
#include "utilities_common.h"

using namespace std;
using namespace cv;

static void doZCA(const vector<Mat> &images, const Size &size, const float epsilon, const bool gcn, const string &id, int seed)
{
	cout << "epsilon " << epsilon << endl;
	ZCA zca(images, size, epsilon, gcn);

	stringstream name;
	name << "zcaWeights" <<(gcn ? "GCN" : "") << id << "_" << size.width << "_" << seed << "_" << images.size() << ".xml";
	zca.Write(name.str().c_str());
}

// returns true if the given 3 channel image is B = G = R
bool isGrayImage(const Mat &img) 
{
    Mat dst;
    Mat bgr[3];
    split( img, bgr );
    absdiff( bgr[0], bgr[1], dst );

    if(countNonZero( dst ))
        return false;

    absdiff( bgr[0], bgr[2], dst );
    return !countNonZero( dst );
}

int main(void)
{
    vector<string> filePaths;
    GetFilePaths("/media/kjaget/AC8612CF86129A42/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/framegrabber", ".png", filePaths);
    GetFilePaths("/media/kjaget/AC8612CF86129A42/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/Framegrabber2", ".png", filePaths, true);
    GetFilePaths("/media/kjaget/AC8612CF86129A42/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/generic", ".png", filePaths, true);
    GetFilePaths("/media/kjaget/AC8612CF86129A42/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/20160210", ".png", filePaths, true);
    GetFilePaths("/media/kjaget/AC8612CF86129A42/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/white_bg", ".png", filePaths, true);
    cout << filePaths.size() << " images!" << endl;
	const int seed = 12345;
	RandomSubImage rsi(RNG(seed), filePaths);
	Mat img; // full image data
	Mat patch; // randomly selected image patch from full image
	vector<Mat> images;
	const int nImgs = 200000;
    for (int nDone = 0; nDone < nImgs; ) 
	{
		img = rsi.get(1.0, 0.05);
		resize(img, patch, Size(24,24));
		// There are grayscale images in the 
		// negatives, but we'll never see one
		// in real life. Exclude those for now
		if (isGrayImage(img))
			continue;
		images.push_back(patch.clone());
		nDone++;
		if (!(nDone % 1000))
			cout << nDone << " image patches extracted" << endl;
	}
	doZCA(images, Size(12,12), 1, true, "nograysepchannelsE10", seed);
	doZCA(images, Size(24,24), 1, true, "nograysepchannelsE10", seed);
	doZCA(images, Size(12,12), 0.1, true, "nograysepchannelsE1", seed);
	doZCA(images, Size(24,24), 0.1, true, "nograysepchannelsE1", seed);
	doZCA(images, Size(12,12), 0.01, true, "nograysepchannelsE01", seed);
	doZCA(images, Size(24,24), 0.01, true, "nograysepchannelsE01", seed);
	doZCA(images, Size(12,12), 0.001, true, "nograysepchannelsE001", seed);
	doZCA(images, Size(24,24), 0.001, true, "nograysepchannelsE001", seed);
	doZCA(images, Size(12,12), 0.0001, true, "nograysepchannelsE0001", seed);
	doZCA(images, Size(24,24), 0.0001, true, "nograysepchannelsE0001", seed);
	doZCA(images, Size(12,12), 0.00001, true, "nograysepchannelsE00001", seed);
	doZCA(images, Size(24,24), 0.00001, true, "nograysepchannelsE00001", seed);
	doZCA(images, Size(12,12), 1, false, "nograysepchannelsE10", seed);
	doZCA(images, Size(24,24), 1, false, "nograysepchannelsE10", seed);
	doZCA(images, Size(12,12), 0.1, false, "nograysepchannelsE1", seed);
	doZCA(images, Size(24,24), 0.1, false, "nograysepchannelsE1", seed);
	doZCA(images, Size(12,12), 0.01, false, "nograysepchannelsE01", seed);
	doZCA(images, Size(24,24), 0.01, false, "nograysepchannelsE01", seed);
	doZCA(images, Size(12,12), 0.001, false, "nograysepchannelsE001", seed);
	doZCA(images, Size(24,24), 0.001, false, "nograysepchannelsE001", seed);
	doZCA(images, Size(12,12), 0.0001, false, "nograysepchannelsE0001", seed);
	doZCA(images, Size(24,24), 0.0001, false, "nograysepchannelsE0001", seed);
	doZCA(images, Size(12,12), 0.00001, false, "nograysepchannelsE00001", seed);
	doZCA(images, Size(24,24), 0.00001, false, "nograysepchannelsE00001", seed);
}
