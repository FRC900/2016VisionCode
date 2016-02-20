/*
 * Main.cpp
 *
 * Created on: Dec 31, 2014
 * Author: jrparks
 */
#include <opencv2/opencv.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include "zedin.hpp"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    ZedIn camera(argc == 2 ? argv[1] : NULL);
    //ZedIn camera(NULL, argc == 2 ? argv[1] : NULL);

    char name[PATH_MAX];
    int  index = 0;
    int  rc;
    struct stat statbuf;
    do
    {
        sprintf(name, "cap%d.avi", index++);
        rc = stat(name, &statbuf);
    } while (rc == 0);
    VideoWriter outputVideo(name, CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(camera.width(), camera.height()), true);
    Mat frame;

    while (true)
    {
        camera.getNextFrame(frame, false);
        if (!frame.empty())
        {
            outputVideo << frame;
        }
        else
        {
            fprintf(stderr, "Unable to grab frame.\n");
            break;
        }
	imshow("Frame", frame);
        uchar wait_key = cv::waitKey(2);
        if ((wait_key == 27) || (wait_key == 32))
        {
            break;
        }
    }
    fprintf(stdout, "Closing camera.\n");
}
