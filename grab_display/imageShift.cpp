#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
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
int main(int argc, char *argv[])
{
    VideoCapture image(argv[1]);
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
    Mat original;
    Mat copy;
    Mat final;
    image >> original;
    int expand = original.rows * dx / 2;
    for (int is = 0; is < 5; is++)
    {
        for (int ix = 0; ix < 3; ix++)
        {
            for (int iy = 0; iy < 3; iy++)
            {
                original.copyTo(copy);
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
                //imshow("Final", final);
                bool truth = imwrite(write_file, final);
                if(truth == false)
                {
                    cout << "Error! Could not write file." << endl;
                }
                //waitKey(1000);
            }
        }
    }
}
