#include <iostream>
#include <opencv2/opencv.hpp>

#include "zedin.hpp"
#include "GoalDetector.hpp"

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

   GoalDetector gd;
   Mat image;
   Mat depth;
   while(true)
   {
	cap->update();
	cap->getFrame().copyTo(image);
	cap->getDepth().copyTo(depth);
	imshow ("Normalized Depth", cap->getNormalDepth());
	imshow ("Image", image);
	
	gd.processFrame(image,depth);
	cout << "Distance to goal: " << gd.dist_to_goal() << endl;
	cout << "Angle to goal: " << gd.angle_to_goal() << endl;

	waitKey(5);
   }
}
