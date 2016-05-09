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
    return Rect(rectIn.tl().x - (dx*rectIn.width/ds), rectIn.tl().y-(dy*rectIn.height/ds), rectIn.width/ds, rectIn.height/ds);
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
    string dir_name = "";
    Mat original = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat copy;
    Mat final;
    int expand = original.rows / 2;
    for (int is = 0; is < 5; is++)
    {
        for (int ix = 0; ix < 2; ix++)
        {
            for (int iy = 0; iy < 2; iy++)
            {
                original.copyTo(copy);
                copyMakeBorder(copy, copy, expand, expand, expand, expand, BORDER_CONSTANT, Scalar(0,0,255));
                Rect ROI = Rect(expand, expand, original.cols, original.rows);
                ROI = shiftRect(ROI, ds[is], (ix-1)*dx, (iy-1)*dy);
                copy(ROI).copyTo(final);
                /*original.copyTo(copy);
                original.copyTo(final);
                dir_name = to_string(is*9 + ix*3 + iy);
                Rect ROI = Rect(0,0,copy.cols,copy.rows);
                if (ix == 0)
                {
                    copyMakeBorder(copy, copy, 0, 0, 0, expand, BORDER_CONSTANT, Scalar(0,0,255));
                    ROI = Rect(Point(expand,0), Point(copy.cols, ROI.br().y));
                }
                if (ix == 2)
                {
                    copyMakeBorder(copy, copy, 0, 0, expand, 0, BORDER_CONSTANT, Scalar(0,0,255));
                }
                if (iy == 0)
                {
                    copyMakeBorder(copy, copy, 0, expand, 0, 0, BORDER_CONSTANT, Scalar(0,0,255));
                    ROI = Rect(Point(ROI.tl().x, expand), Point(ROI.br().x, copy.rows));
                }
                if (iy == 2)
                {
                    copyMakeBorder(copy, copy, expand, 0, 0, 0, BORDER_CONSTANT, Scalar(0,0,255));
                }
                Mat copy1;
                copy1 = copy(ROI).clone();
                Mat copy2;
                if (is < 2)
                {
                    int resize_val = (copy1.rows - copy1.rows * sqrt(ds[is]));
                    copyMakeBorder(copy1, copy1, resize_val, resize_val, resize_val, resize_val, BORDER_CONSTANT, Scalar(0,0,255));
                    resize(copy1, final, final.size(), 0, 0, INTER_NEAREST);
                }
                else if (is > 2)
                {
                    int resize_val = (copy1.rows * sqrt(ds[is]) - copy1.rows);
                    ROI = Rect(Point(resize_val, resize_val), Point(copy1.cols - resize_val, copy1.rows - resize_val));
                    copy2 = copy1(ROI).clone();
                    resize(copy2, final, final.size(), 0, 0, INTER_NEAREST);
                }
                else
                {
                    copy1.copyTo(final);
                }
                */
                dir_name = to_string(is*9 + ix*3 + iy);
				if (mkdir((output_dir+"/"+dir_name).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH))
				{
					if (errno != EEXIST)
					{
						cerr << "Could not create " << (output_dir+"/"+dir_name).c_str() << ":";
						perror("");
					}
				}

                string write_file = output_dir + "/" + dir_name + "/" + filename;
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
