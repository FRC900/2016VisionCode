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
#include "random_subimage.hpp"

using namespace std;
using namespace cv;

static Rect shiftRect(const Rect rectIn, float ds, float dx, float dy)
{
	return Rect(cvRound(rectIn.tl().x - (dx*rectIn.width /ds)), 
			cvRound(rectIn.tl().y - (dy*rectIn.height/ds)), 
			cvRound(rectIn.width /ds),
			cvRound(rectIn.height/ds));
}

// Create the various output dirs - the base shift
// directory and directories numbered 0 - 44.
void createShiftDirs(const string &outputDir)
{
	if (mkdir((outputDir).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH))
	{
		if (errno != EEXIST)
		{
			cerr << "Could not create " << outputDir.c_str() << ":";
			perror("");
			return;
		}
	}
	// Create output directories
	for (int is = 0; is < 5; is++)
	{
		for (int ix = 0; ix <= 2; ix++)
		{
			for (int iy = 0; iy <= 2; iy++)
			{
				string dir_name = to_string(is*9 + ix*3 + iy);
				if (mkdir((outputDir+"/"+dir_name).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH))
				{
					if (errno != EEXIST)
					{
						cerr << "Could not create " << (outputDir+"/"+dir_name).c_str() << ":";
						perror("");
					}
				}
			}
		}
	}
}


// Given a src image and an object ROI within that image,
// generate shifted versions of the object
// maxRot is in radians.
void doShifts(const Mat &src, const Rect &objROI, RNG &rng, const Point3f &maxRot, int copiesPerShift, const string &outputDir, const string &fileName)
{
	Mat rotImg;  // randomly rotated input
	Mat rotMask; // and mask
	Mat final;   // final output

	// x, y and size shift values
	const float dx = .17;
	const float dy = .17;
	const float ds[5] = {.83, .91, 1.0, 1.10, 1.21};

	if (src.empty())
	{
		return;
	}

	// strip off directory and .png suffix
	string fn(fileName);
	auto pos = fn.rfind('/');
	if (pos != std::string::npos)
	{
		fn.erase(0, pos + 1);
	}
	pos = fn.rfind('.');
	if (pos != std::string::npos)
	{
		fn.erase(pos);
	}
	cout << fn << endl;

	const Scalar fillColor = Scalar(src(objROI).at<Vec3b>(0,0));
	// create another rect expanded to the limits of the input
	// image size with the object still in the center.
	// This will allow us to save the pixels from a corner as
	// the image is rotated
	const double targetAR = (double) objROI.width / objROI.height;
	const int added_x = min(objROI.tl().x, src.cols - 1 - objROI.br().x);
	const int added_y = min(objROI.tl().y, src.rows - 1 - objROI.br().y);
	const int added_size = min(added_x, int(added_y * targetAR));
	const Rect largeRect(objROI.tl() - Point(added_size, added_size/targetAR),
				   objROI.size() + Size(2*added_size, 2*int(added_size/targetAR)));

	const Rect largeRectBounds(0,0,largeRect.width, largeRect.height);
	// This is a rect which will be the input objROI but
	// in coorindates relative to the largeRect created above
	const Rect newObjROI(added_size, added_size/targetAR, objROI.width, objROI.height);
	// Generate copiesPerShift images per shift/scale permutation
	// So each call will end up with 5 * 3 * 3 * copiesPerShift
	// images writen
	for (int is = 0; is < 5; is++)
	{
		for (int ix = 0; ix <= 2; ix++)
		{
			for (int iy = 0; iy <= 2; iy++)
			{
				for (int c = 0; c < copiesPerShift; c++)
				{
					// Shift/rescale the region of interest based on
					// which permuation of the shifts/rescales we're at
					const Rect thisROI = shiftRect(newObjROI, ds[is], (ix-1)*dx, (iy-1)*dy);
					if ((largeRectBounds & thisROI) != thisROI)
					{
						cerr << "Rectangle out of bounds for " << is 
							<< " " << ix << " " << iy << " " << 
							largeRectBounds.size() << " vs " << thisROI << endl;
						break;
					}

					// Rotate the image a random amount.  Mask isn't used
					// since there's no chroma-keying going on.
					rotateImageAndMask(src(largeRect), Mat(), fillColor, maxRot, rng, rotImg, rotMask);

					rotImg(thisROI).copyTo(final);
#if 0
					imshow("src", src);
					imshow("src(objROI)", src(objROI));
					imshow("src twice ROI", src(twiceObjROI));
					imshow("src(twice ROI)(newObjROI)", src(twiceObjROI)(newObjROI));
					imshow("rotImg", rotImg);
					imshow("rotImg(newObjROI)", rotImg(newObjROI));
					resize (final, final, Size(240,240));
					imshow("Final", final);
					waitKey(0);
#else
					// 48x48 is the largest size we'll need from here on out,
					// so resize to that to save disk space
					resize (final, final, Size(48,48));
#endif

					// Dir name is a number from 0 - 44.
					// 1 per permutation of x,y shift plus resize
					string dir_name = to_string(is*9 + ix*3 + iy);
					string write_file = outputDir + "/" + dir_name + "/" + fn + "_" + to_string(c) + ".png";
					if (imwrite(write_file, final) == false)
					{
						cout << "Error! Could not write file "<<  write_file << endl;
					}
				}
			}
		}
	}
}


void doShifts(const Mat &src, const Mat &mask, RNG &rng, RandomSubImage &rsi, const Point3f &maxRot, int copiesPerShift, const string &outputDir, const string &fileName)
{
	Mat original;
	Mat objMask;
	Mat bgImg;    // random background image to superimpose each input onto 
	Mat chromaImg; // combined input plus bg
	Mat rotImg;  // randomly rotated input
	Mat rotMask; // and mask
	Mat final;   // final output

	// x, y and size shift values
	const float dx = .17;
	const float dy = .17;
	const float ds[5] = {.83, .91, 1.0, 1.10, 1.21};

	if (src.empty() || mask.empty())
	{
		return;
	}

	// strip off directory and .png suffix
	string fn(fileName);
	auto pos = fn.rfind('/');
	if (pos != std::string::npos)
	{
		fn.erase(0, pos + 1);
	}
	pos = fn.rfind('.');
	if (pos != std::string::npos)
	{
		fn.erase(pos);
	}
	cout << fn << endl;

	// Use color at 0,0 to fill in expanded rect assuming that
	// location is the chroma-key color for that given image
	// Probably want to pass this in instead for cases
	// where we're working from a list of files captured from live
	// video rather than video shot against a fixed background - can't
	// guarantee the border color there is safe to use
	const Scalar fillColor = Scalar(src.at<Vec3b>(0,0));

	// Enlarge the original image.  Since we're shifting the region
	// of interest need to do this to make sure we don't end up 
	// outside the mat boundries
	const int expand = max(src.rows, src.cols) / 2;
	const Rect origROI(expand, expand, src.cols, src.rows);
	copyMakeBorder(src, original, expand, expand, expand, expand, BORDER_CONSTANT, Scalar(fillColor));
	copyMakeBorder(mask, objMask, expand, expand, expand, expand, BORDER_CONSTANT, Scalar(0));
	
	// Generate copiesPerShift images per shift/scale permutation
	// So each call will end up with 5 * 3 * 3 * copiesPerShift
	// images writen
	for (int is = 0; is < 5; is++)
	{
		for (int ix = 0; ix <= 2; ix++)
		{
			for (int iy = 0; iy <= 2; iy++)
			{
				for (int c = 0; c < copiesPerShift; c++)
				{
					// Rotate the image a random amount. Also rotat the mask
					// so they stay in sync with each other
					rotateImageAndMask(original, objMask, fillColor, maxRot, rng, rotImg, rotMask);

					// Get a random background image, superimpose
					// the object on top of that image
					bgImg = rsi.get((double)original.cols / original.rows, 0.05);
					chromaImg = doChromaKey(rotImg, bgImg, rotMask);

					// Shift/rescale the region of interest based on
					// which permuation of the shifts/rescales we're at
					const Rect ROI = shiftRect(origROI, ds[is], (ix-1)*dx, (iy-1)*dy);
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
					// 48x48 is the largest size we'll need from here on out,
					// so resize to that to save disk space
					resize (final, final, Size(48,48));
#endif

					// Dir name is a number from 0 - 44.
					// 1 per permutation of x,y shift plus resize
					string dir_name = to_string(is*9 + ix*3 + iy);
					string write_file = outputDir + "/" + dir_name + "/" + fn + "_" + to_string(c) + ".png";
					if (imwrite(write_file, final) == false)
					{
						cout << "Error! Could not write file "<<  write_file << endl;
					}
				}
			}
		}
	}
}
