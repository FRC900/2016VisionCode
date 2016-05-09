#include <opencv2/opencv.hpp>

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

using namespace std;
using namespace cv;
RNG rng(12345);
Rect shiftRect(const Rect rectIn, float ds, float dx, float dy)
{
    return Rect(cvRound(rectIn.tl().x - (dx*rectIn.width /ds)), 
			    cvRound(rectIn.tl().y - (dy*rectIn.height/ds)), 
				cvRound(rectIn.width /ds),
				cvRound(rectIn.height/ds));
}
int main(int argc, char *argv[])
{
    string filename = argv[1];
    auto pos = filename.rfind('/');
    if (pos != std::string::npos)
    {
        filename.erase(0, pos + 1);
    }
    string output_dir = argv[2];
    Vec3b red = Vec3b(0,0,255);
    float dx = .17;
	float dy = .17;
	float ds[5] = {.83, .91, 1.0, 1.10, 1.21};
	Mat original = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	int expand = original.rows / 2;
	Rect origROI(expand, expand, original.cols, original.rows);
	copyMakeBorder(original, original, expand, expand, expand, expand, BORDER_CONSTANT, Scalar(0,0,255));
    Mat final;
    for (int is = 0; is < 5; is++)
    {
        for (int ix = 0; ix <= 2; ix++)
        {
            for (int iy = 0; iy <= 2; iy++)
            {
                Rect ROI = shiftRect(origROI, ds[is], (ix-1)*dx, (iy-1)*dy);
                original(ROI).copyTo(final);

                string dir_name = to_string(is*9 + ix*3 + iy);
				if (mkdir((output_dir+"/"+dir_name).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH))
				{
					if (errno != EEXIST)
					{
						cerr << "Could not create " << (output_dir+"/"+dir_name).c_str() << ":";
						perror("");
					}
				}

                for(int i = 0; i < final.rows; i++)
                {
                    for(int j = 0; j < final.cols; j++)
                    {
                        if(final.at<Vec3b>(i,j) == red)
                        {
                            final.at<Vec3b>(i,j) = Vec3b(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
                        }
                    }
                }
				resize (final, final, Size(24,24));
                //imshow("Final", final);
                string write_file = output_dir + "/" + dir_name + "/" + filename;
                bool truth = imwrite(write_file, final);
                if(truth == false)
                {
                    cout << "Error! Could not write file "<<  write_file << endl;
                }
                //waitKey(1000);
            }
        }
    }
}
