#include <iostream>
#include <opencv2/opencv.hpp>

#include "zedin.hpp"
#include "GoalDetector.hpp"
#include "Utilities.hpp"
#include "track3d.hpp"
#include "frameticker.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
	ZedIn cap(argc == 2 ? argv[1] : NULL, NULL, true);

	const float HFOV =  105 * M_PI / 180.;
	GoalDetector gd(Point2f(HFOV, HFOV * 720. / 1280.), Size(cap.width(),cap.height()));
	gd._draw = true;

	namedWindow("RangeControl", WINDOW_AUTOSIZE);

	createTrackbar("HueMin","RangeControl", &gd._hue_min, 179);
	createTrackbar("HueMax","RangeControl", &gd._hue_max, 179);

	createTrackbar("SatMin","RangeControl", &gd._sat_min, 255);
	createTrackbar("SatMax","RangeControl", &gd._sat_max, 255);

	createTrackbar("ValMin","RangeControl", &gd._val_min, 255);
	createTrackbar("ValMax","RangeControl", &gd._val_max, 255);

	Mat image;
	Mat depth;
	Mat depthNorm;
	Rect bound;
	FrameTicker frameTicker;
	while(cap.getNextFrame(image, false))
	{
		cap.getDepthMat(depth);
		cap.getNormDepthMat(depthNorm);
		frameTicker.mark();
		imshow ("Normalized Depth", depthNorm);

		gd.processFrame(image,depth,bound);
		cout << "Distance to goal: " << gd.dist_to_goal() << endl;
		cout << "Angle to goal: " << gd.angle_to_goal() << endl;
		stringstream ss;
		ss << fixed << setprecision(2) << frameTicker.getFPS() << "FPS";
		putText(image, ss.str(), Point(image.cols - 15 * ss.str().length(), 50), FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,255));
		imshow ("Image", image);

		if ((uchar)waitKey(5) == 27)
		{
			break;
		}
	}
}
