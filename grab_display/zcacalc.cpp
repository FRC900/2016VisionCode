#include <string>
#include "zca.hpp"

#include "utilities_common.h"

using namespace std;
using namespace cv;

int main(void)
{
    vector<string> filePaths;
    GetFilePaths("/media/kjaget/84CA3305CA32F2D2/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/framegrabber", ".png", filePaths);
    GetFilePaths("/media/kjaget/84CA3305CA32F2D2/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/Framegrabber2", ".png", filePaths, true);
    GetFilePaths("/media/kjaget/84CA3305CA32F2D2/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/generic", ".png", filePaths, true);
    cout << filePaths.size() << " images!" << endl;
	const int seed = 12345;
	RNG rng(seed);
	vector<Mat> images;

	const int nImgs = 100000;
	Mat img; // full image data
	Mat patch; // randomly selected image patch from full image

    for (int nDone = 0; nDone < nImgs; ) 
	{
		// Grab a random image from the list
		size_t idx = rng.uniform(0, filePaths.size());
		img = imread(filePaths[idx]);

		// Pick a random row and column from the image
		int r = rng.uniform(0, img.rows);
		int c = rng.uniform(0, img.cols);

		// Pick a random size as well. Make sure it
		// doesn't extend past the edge of the input
		int s = rng.uniform(0, 2*MIN(MIN(r, img.rows-r), MIN(c, img.cols-c)) + 1 );

		if (s < 28)
			continue;

		Rect rect = Rect(c-s/2, r-s/2, s, s);
		//cout << "Using " << filePaths[idx] << rect << endl;
		// Resize patches to 24x24 to save memory
		resize(img(rect), patch, Size(24,24));
		images.push_back(patch.clone());
		nDone++;
		if (!(nDone % 1000))
			cout << nDone << " image patches extracted" << endl;
	}

	ZCA zca12(images, Size(12,12));
	ZCA zca24(images, Size(24,24));

	stringstream name;
	name << "zcaWeights12_" << seed << "_" << nImgs << ".xml";
	zca12.Write(name.str().c_str());
	name.str(string());
	name << "zcaWeights24_" << seed << "_" << nImgs << ".xml";
	zca24.Write(name.str().c_str());
}
