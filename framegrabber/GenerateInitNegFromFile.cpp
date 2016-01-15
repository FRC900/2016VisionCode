#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "utilities_common.h"

using namespace std;
using namespace cv;

int main(int argc, const char* argv[])
{
    const string oFolder = "/home/kjaget/CNN/negative";
    
    vector<string> filePaths;
    GetFilePaths("/media/kjaget/84CA3305CA32F2D2/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/framegrabber", ".png", filePaths);
    GetFilePaths("/media/kjaget/84CA3305CA32F2D2/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/Framegrabber2", ".png", filePaths, true);
    GetFilePaths("/media/kjaget/84CA3305CA32F2D2/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/generic", ".png", filePaths, true);
    cout << filePaths.size() << " images!" << endl;

    RNG rng( 0xFFFFFFFE );
    rng.uniform( 0, 100 );
    const int nNegs = 200000 + 10000; // 10000 for validation

    Mat img;
    Mat patch;
    Mat rsz;
    
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    for (int nDone = 0; nDone < nNegs; ) {
        size_t idx = rng.uniform(0, filePaths.size());
	img = imread(filePaths[idx]);
	
        int r = rng.uniform(0, img.rows);
        int c = rng.uniform(0, img.cols);
        int s = rng.uniform(0, 2*MIN(MIN(r, img.rows-r), MIN(c, img.cols-c)) + 1 );
        
        if (s < 28)
            continue;
        
        Rect rect = Rect(c-s/2, r-s/2, s, s);
        patch = img(rect);
        cv::resize(patch, rsz, cv::Size(48, 48));
        stringstream ss;
        ss << oFolder << "/" << filePaths[idx].substr(filePaths[idx].find_last_of("\\/") + 1) << "_" << setfill('0') << "_" << setw(4) << rect.x << "_" << setw(4) << rect.y << "_" << setw(4) << rect.width << "_" << setw(4) << rect.height << ".png";
	cerr << ss.str() << endl;
        imwrite(ss.str(), rsz, compression_params);
	nDone++;
        if (nDone % 1000 == 0)
            cout << nDone << " neg generated!" << endl;
    }
    
    return 0;
}
