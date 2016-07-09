#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#include "chroma_key.hpp"
#include "image_warp.hpp"

using namespace std;
using namespace cv;

#if 1
//Values for purple screen:
static int g_h_max = 170;
static int g_h_min = 130;
static int g_s_max = 255;
static int g_s_min = 147;
static int g_v_max = 255;
static int g_v_min = 48;
#else
//Values for blue screen:
static int g_h_max = 120;
static int g_h_min = 110;
static int g_s_max = 255;
static int g_s_min = 220;
static int g_v_max = 150;
static int g_v_min = 50;
#endif

static Rect shiftRect(const Rect rectIn, float ds, float dx, float dy)
{
	return Rect(cvRound(rectIn.tl().x - (dx*rectIn.width /ds)), 
			cvRound(rectIn.tl().y - (dy*rectIn.height/ds)), 
			cvRound(rectIn.width /ds),
			cvRound(rectIn.height/ds));
}

int main(int argc, char *argv[])
{
	RNG rng(12345);
	const int bgCount = 10;

	// bgfile has a list of backgrounds
	// to superimpose the shifted image
	// onto
	ifstream bgfile(argv[2]);
	string bgfilename;
	vector<string> bgFileList;
	while (getline(bgfile, bgfilename))
	{
		bgFileList.push_back(bgfilename);
	}
	bgfile.close();

	string output_dir = argv[3];

	Mat original; // input image
	Mat objMask;  // binary mask for chroma-keyin
	Mat bgImg;    // random background image to superimpose the input onto 
	Mat chromaImg; // combined input plus bg
	Mat rotImg;  // randomly rotated input
	Mat rotMask; // and mask
	Mat final;   // final output

	// x, y and size shift values
	const float dx = .17;
	const float dy = .17;
	const float ds[5] = {.83, .91, 1.0, 1.10, 1.21};

	// Create output directories
	for (int is = 0; is < 5; is++)
	{
		for (int ix = 0; ix <= 2; ix++)
		{
			for (int iy = 0; iy <= 2; iy++)
			{
				string dir_name = to_string(is*9 + ix*3 + iy);
				if (mkdir((output_dir+"/"+dir_name).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH))
				{
					if (errno != EEXIST)
					{
						cerr << "Could not create " << (output_dir+"/"+dir_name).c_str() << ":";
						perror("");
					}
				}
			}
		}
	}

	ifstream filelist(argv[1]);
	string filename;
	// Grab next filename from list in input, 
	while (getline(filelist, filename))
	{
		original = imread(filename, CV_LOAD_IMAGE_COLOR);
		if (original.empty())
		{
			cerr << "Could not read input image " << filename <<endl;
			continue;
		}

		// strip off directory and .png suffix
		auto pos = filename.rfind('/');
		if (pos != std::string::npos)
		{
			filename.erase(0, pos + 1);
		}
		pos = filename.rfind('.');
		if (pos != std::string::npos)
		{
			filename.erase(pos);
		}

		// Use color at 0,0 to fill in expanded rect assuming that
		// location is the chroma-key color for that given image
		Vec3b fillColor = original.at<Vec3b>(0,0); 
		int expand = max(original.rows, original.cols) / 2;
		Rect origROI(expand, expand, original.cols, original.rows);
		copyMakeBorder(original, original, expand, expand, expand, expand, BORDER_CONSTANT, Scalar(fillColor));
		Mat hsvframe;
		cvtColor(original, hsvframe, CV_BGR2HSV);
		Rect boundingRect; // notused
		if (!getMask(hsvframe, Scalar(g_h_min, g_s_min, g_v_min), Scalar(g_h_max, g_s_max, g_v_max), objMask, boundingRect))
		{
			cerr << "Could not find object in frame" << endl;
			return -1;
		}
		// Only need the mask from here on out
		hsvframe.release();

		for (int is = 0; is < 5; is++)
		{
			for (int ix = 0; ix <= 2; ix++)
			{
				for (int iy = 0; iy <= 2; iy++)
				{
					for (int bg = 0; bg < bgCount; bg++)
					{
						bgImg = randomSubImage(rng, bgFileList, (double)original.cols / original.rows, 0.05);
						const Scalar bgColor(fillColor);
						rotateImageAndMask(original, objMask, bgColor, 
								      Point3f(0,0,M_PI*2.), rng, rotImg, rotMask);

						chromaImg = doChromaKey(rotImg, bgImg, rotMask);

						Rect ROI = shiftRect(origROI, ds[is], (ix-1)*dx, (iy-1)*dy);
						chromaImg(ROI).copyTo(final);
#if 0
						imshow("original", original);
						imshow("bgImg", bgImg);
						imshow("rotImg", rotImg);
						imshow("rotMask", rotMask);
						imshow("chromaImg", chromaImg);
						resize (final, final, Size(240,240));
						imshow("Final", final);
						waitKey(0);
#else
						resize (final, final, Size(24,24));
#endif

						string dir_name = to_string(is*9 + ix*3 + iy);
						string write_file = output_dir + "/" + dir_name + "/" + filename + "_" + to_string(bg) +".png";
						if (imwrite(write_file, final) == false)
						{
							cout << "Error! Could not write file "<<  write_file << endl;
						}
					}
				}
			}
		}
	}
}
