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

   GoalDetector gd(Point2f(84.14 * (M_PI / 180.0), 53.836 * (M_PI / 180.0)), Size(cap->width(),cap->height()));
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

	waitKey(5);
   }
}
