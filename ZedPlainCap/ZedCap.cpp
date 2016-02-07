/*
 * Main.cpp
 *
 * Created on: Dec 31, 2014
 * Author: jrparks
 */
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include "zedin.cpp"

using namespace std;
using namespace cv;


int main(int argc, char *argv[])
{
    ZedIn *camera = NULL;

    if (argc == 2)
    {
        camera = new ZedIn(argv[1]);
    }
    else
    {
        camera = new ZedIn;
    }
    char name[PATH_MAX];
    int  index = 0;
    int  rc;
    struct stat statbuf;
    do
    {
        sprintf(name, "cap%d.avi", index++);
        rc = stat(name, &statbuf);
    } while (rc == 0);
    VideoWriter outputVideo(name, CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(camera->width(), camera->height()), true);
    Mat frame(200,200,CV_8UC3);

    imshow("Frame", frame);

    while (true)
    {
        camera->update();
        camera->getFrame().copyTo(frame);
        if (!frame.empty())
        {
            outputVideo << frame;
        }
        else
        {
            fprintf(stderr, "Unable to grab frame.\n");
            break;
        }
        uchar wait_key = cv::waitKey(5);
	cout << wait_key << endl;
        if ((wait_key == 27) || (wait_key == 32))
        {
            break;
        }
    }
    fprintf(stdout, "Closing camera.\n");
}
