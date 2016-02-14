#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <stdlib.h>
#include "Args.hpp"

using namespace std;

static void Usage(void)
{
   cout << "Usage : test [option list] [camera # | video name]" << endl << endl;
   cout << "  Camera number is an integer corresponding to /dev/video??" << endl;

   cout << "\t--frame=<frame num>  start at given frame" << endl;
   cout << "\t--all                write to disk all detected images in each frame" << endl;
   cout << "\t--batch              run without GUI" << endl;
   cout << "\t--skip=<x>           skip frames, only processing every <x> - file input only." << endl;
   cout << "\t--calibrate          bring up crosshair to calibrate camera position" << endl;
   cout << "\t--capture            save camera video to output file" << endl;
   cout << "\t--save               write processed video to output file" << endl;
   cout << "\t--no-rects           start with detection rectangles disabled" << endl;
   cout << "\t--no-tracking        start with tracking rectangles disabled" << endl;
   cout << "\t--classifierBase=    base directory for classifier info" << endl;
   cout << "\t--classifierDir=     pick classifier dir and stage number" << endl;
   cout << "\t--classifierStage=   from command line" << endl;
   cout << "\t--groundTruth        only test frames which have ground truth data " << endl;
   cout << endl;
   cout << "Examples:" << endl;
   cout << "test : start in GUI mode, open default camera, start detecting and tracking while displaying results in the GUI" << endl;
   cout << "test --batch --capture : Start without a gui, write captured video to disk.  This is the normal way the code is run on the robot during matches. The code communicates via network tables to the roborio." << endl;
   cout << "test --batch --all foo.mp4 : run through foo.mp4. For each frame, write to disk images of all recognized objects. Might be useful for grabbing negative samples from a video known to have no positives in it" << endl;
}

Args::Args(void)
{
	captureAll         = false;
	tracking           = true;
	rects              = true;
	batchMode          = false;
	skip               = 0;
	calibrate          = false;
	writeVideo         = false;
	saveVideo          = false;
	classifierBaseDir  = "/home/ubuntu/2016VisionCode/bindetection/d";
	classifierDirNum   = 0;
	classifierStageNum = 0;
	frameStart         = 0.0;
	groundTruth        = false;
}

bool Args::processArgs(int argc, const char **argv)
{
	const string frameOpt           = "--frame=";          // start at given frame
	const string captureAllOpt      = "--all";             // capture all detected images in each frame
	const string batchModeOpt       = "--batch";           // run without GUI
	const string skipOpt            = "--skip=";           // skip frames in input video file
	const string calibrateOpt       = "--calibrate";       // bring up crosshair to calibrate camera position
	const string writeVideoOpt      = "--capture";         // save camera video to output file
	const string saveVideoOpt       = "--save";            // write processed video to output file
	const string rectsOpt           = "--no-rects";        // start with detection rectangles disabled
	const string trackingOpt        = "--no-tracking";     // start with tracking rectangles disabled
	const string classifierBaseOpt  = "--classifierBase="; // classifier base dir
	const string classifierDirOpt   = "--classifierDir=";  // pick classifier dir and stage number
	const string classifierStageOpt = "--classifierStage=";// from command line
	const string groundTruthOpt     = "--groundTruth";     // only test frames which have ground truth data
	const string badOpt             = "--";
	// Read through command line args, extract
	// cmd line parameters and input filename
	int fileArgc;
	for (fileArgc = 1; fileArgc < argc; fileArgc++)
	{
		if (frameOpt.compare(0, frameOpt.length(), argv[fileArgc], frameOpt.length()) == 0)
			frameStart = atoi(argv[fileArgc] + frameOpt.length());
		else if (captureAllOpt.compare(0, captureAllOpt.length(), argv[fileArgc], captureAllOpt.length()) == 0)
			captureAll = true;
		else if (batchModeOpt.compare(0, batchModeOpt.length(), argv[fileArgc], batchModeOpt.length()) == 0)
			batchMode = true;
		else if (skipOpt.compare(0, skipOpt.length(), argv[fileArgc], skipOpt.length()) == 0)
			skip = atoi(argv[fileArgc] + skipOpt.length());
		else if (calibrateOpt.compare(0, calibrateOpt.length(), argv[fileArgc], calibrateOpt.length()) == 0)
			calibrate = true;
		else if (writeVideoOpt.compare(0, writeVideoOpt.length(), argv[fileArgc], writeVideoOpt.length()) == 0)
			writeVideo = true;
		else if (saveVideoOpt.compare(0, saveVideoOpt.length(), argv[fileArgc], saveVideoOpt.length()) == 0)
			saveVideo = true;
		else if (trackingOpt.compare(0, trackingOpt.length(), argv[fileArgc], trackingOpt.length()) == 0)
			tracking = false;
		else if (rectsOpt.compare(0, rectsOpt.length(), argv[fileArgc], rectsOpt.length()) == 0)
			rects = false;
		else if (classifierBaseOpt.compare(0, classifierBaseOpt.length(), argv[fileArgc], classifierBaseOpt.length()) == 0)
			classifierBaseDir = string(argv[fileArgc] + classifierBaseOpt.length());
		else if (classifierDirOpt.compare(0, classifierDirOpt.length(), argv[fileArgc], classifierDirOpt.length()) == 0)
			classifierDirNum = atoi(argv[fileArgc] + classifierDirOpt.length());
		else if (classifierStageOpt.compare(0, classifierStageOpt.length(), argv[fileArgc], classifierStageOpt.length()) == 0)
			classifierStageNum = atoi(argv[fileArgc] + classifierStageOpt.length());
		else if (groundTruthOpt.compare(0, groundTruthOpt.length(), argv[fileArgc], groundTruthOpt.length()) == 0)
			groundTruth = true;
		else if (badOpt.compare(0, badOpt.length(), argv[fileArgc], badOpt.length()) == 0) // unknown option
		{
			cerr << "Unknown command line option " << argv[fileArgc] << endl;
			Usage();
			return false;
		}
		else // first non -- arg is filename or camera number
			break;
	}
	if (argc > (fileArgc + 1))
	{
	   cerr << "Extra arguments after file name" << endl;
	   Usage();
	   return false;
	}
	if (fileArgc < argc)
		inputName = argv[fileArgc];
	return true;
}
