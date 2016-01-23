#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#include <cstdio>
#include <ctime>

#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <boost/thread.hpp>

#include "zedin.hpp"

using namespace std;
using namespace cv;

bool leftCamera = true;


int main(int argc, char **argv) {

  ZedIn *cap = NULL;
  if(argc == 2) {
  	cap = new ZedIn(argv[1]);
	cout << "Read SVO file" << endl;
  }
  else {
	cap = new ZedIn;
	cout << "Initialized camera" << endl;
  }
  
  
  Mat frame;
  clock_t startTime;
  while(1)
    {
    startTime = clock();
    cap->update();
    cap->getFrame().copyTo(frame);
    imshow("frame",frame);
    std::cout << "Took: " << (((double)clock() - startTime) / CLOCKS_PER_SEC) << " seconds" << endl;
    waitKey(5);
    }
}
