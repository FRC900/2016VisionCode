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

#include "FovisLocalizer.hpp"

#include "kalman/ekfilter.hpp"

using namespace std;
using namespace cv;


int main(int argc, char **argv) {

	ZedIn *cap = NULL;
	if(argc == 2) {
		cap = new ZedIn(argv[1]);
		cerr << "Read SVO file" << endl;
	}
	else {
		cap = new ZedIn();
		cerr << "Initialized camera" << endl;
	}

	Mat frame, depthFrame;

	clock_t startTime;

	cap->update();
	cap->getFrame().copyTo(frame);
	
	FovisLocalizer fvlc(cap->getCameraParams(), cap->width(), cap->height(), frame);

	while(1)
	{

		startTime = clock();

		cap->update();
		cap->getFrame().copyTo(frame); //pull frame from zed
		cap->getDepth().copyTo(depthFrame);

		fvlc.processFrame(frame,depthFrame);

		imshow("frame",frame);

		cout << "XYZ " << fvlc.getTransform().first << endl;
		cout << "RPY " << fvlc.getTransform().second << endl;

		waitKey(5);
	}
}
