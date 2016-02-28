#include <iostream>
#include <opencv2/opencv.hpp>

#include "zedin.hpp"
#include "GoalDetector.hpp"
#include "Utilities.hpp"
#include "track3d.hpp"

using namespace cv;
using namespace std;


int main(int argc, char **argv)
{
   ZedIn *cap = NULL;
   if(argc == 2) {
  	cap = new ZedIn(argv[1]);
	cerr << "Read SVO file" << endl;
   }
   else {
	cap = new ZedIn;
	cerr << "Initialized camera" << endl;
   }

   cap->left(true);

const float HFOV =  51.3 * M_PI / 180.;
   GoalDetector gd(Point2f(HFOV, HFOV * 480. / 640.), Size(cap->width(),cap->height()));
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
   Rect bound;
   while(true)
   {
	cap->update();
	cap->getFrame().copyTo(image);
	cap->getDepth().copyTo(depth);
	imshow ("Normalized Depth", cap->getNormalDepth());
	
	gd.processFrame(image,depth,bound);
	cout << "Distance to goal: " << gd.dist_to_goal() << endl;
	cout << "Angle to goal: " << gd.angle_to_goal() << endl;
	imshow ("Image", image);

	if ((uchar)waitKey(5) == 27)
	{
		break;
	}
   }
}
