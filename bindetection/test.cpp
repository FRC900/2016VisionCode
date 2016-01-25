#include <iostream>
#include <iomanip>
#include <cstdio>
#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "networktables/NetworkTable.h"
#include "networktables2/type/NumberArray.h"

#include "classifierio.hpp"
#include "detectstate.hpp"
#include "frameticker.hpp"
#include "groundtruth.hpp"
#include "objdetect.hpp"
#include "videoin.hpp"
#include "imagein.hpp"
#include "camerain.hpp"
#include "c920camerain.hpp"
#include "zedin.hpp"
#include "track.hpp"
#include "Args.hpp"
#include "WriteOnFrame.hpp"

using namespace std;
using namespace cv;

//function prototypes
void writeImage(const Mat &frame, const vector<Rect> &rects, size_t index, const char *path, int frameCounter);
string getDateTimeString(void);
void writeNetTableBoolean(NetworkTable *netTable, string label, int index, bool value);
void drawRects(Mat image ,vector<Rect> detectRects, Scalar rectColor = Scalar(0,0,255), bool text = true);
void drawTrackingInfo(Mat &frame, vector<TrackedObjectDisplay> &displayList);
void checkDuplicate (vector<Rect> detectRects);
void openMedia(const string &fileName, MediaIn *&cap, string &capPath, string &windowName, bool gui);
void openVideoCap(const string &fileName, VideoIn *&cap, string &capPath, string &windowName, bool gui);
string getVideoOutName(bool raw = true);
void writeVideoToFile(VideoWriter &outputVideo, const char *filename, const Mat &frame, NetworkTable *netTable, bool dateAndTime);

void drawRects(Mat image, vector<Rect> detectRects, Scalar rectColor, bool text)
{
    for(vector<Rect>::const_iterator it = detectRects.begin(); it != detectRects.end(); ++it)
	{
		// Mark detected rectangle on image
		// Change color based on direction we think the bin is pointing
	    rectangle(image, *it, rectColor, 3);
		// Label each outlined image with a digit.  Top-level code allows
		// users to save these small images by hitting the key they're labeled with
		// This should be a quick way to grab lots of falsly detected images
		// which need to be added to the negative list for the next
		// pass of classifier training.
		if (text)
		{
			size_t i = it - detectRects.begin();
			if (i < 10)
			{
				stringstream label;
				label << i;
				putText(image, label.str(), Point(it->x+10, it->y+30), FONT_HERSHEY_PLAIN, 2.0, rectColor);
			}
		}
	}
}

void drawTrackingInfo(Mat &frame, vector<TrackedObjectDisplay> &displayList)
{
   for (vector<TrackedObjectDisplay>::const_iterator it = displayList.begin(); it != displayList.end(); ++it)
   {
	  if (it->ratio >= 0.15)
	  {
		 const int roundAngTo = 2;
		 const int roundDistTo = 2;

		 // Color moves from red to green (via brown, yuck)
		 // as the detected ratio goes up
		 Scalar rectColor(0, 255 * it->ratio, 255 * (1.0 - it->ratio));
		 // Highlight detected target
		 rectangle(frame, it->rect, rectColor, 3);
		 // Write detect ID, distance and angle data
		 putText(frame, it->id, Point(it->rect.x+25, it->rect.y+30), FONT_HERSHEY_PLAIN, 2.0, rectColor);
		 stringstream distLabel;
		 distLabel << "D=" << fixed << setprecision(roundDistTo) << it->distance;
		 putText(frame, distLabel.str(), Point(it->rect.x+10, it->rect.y-10), FONT_HERSHEY_PLAIN, 1.2, rectColor);
		 stringstream angleLabel;
		 angleLabel << "A=" << fixed << setprecision(roundAngTo) << it->angle;
		 putText(frame, angleLabel.str(), Point(it->rect.x+10, it->rect.y+it->rect.height+20), FONT_HERSHEY_PLAIN, 1.2, rectColor);
	  }
   }
}

void checkDuplicate (vector<Rect> detectRects) {
	for( size_t i = 0; i < detectRects.size(); i++ ) {
		for (size_t j = 0; j < detectRects.size(); j++) {
			if (i != j) {
				Rect intersection = detectRects[i] & detectRects[j];
				if (intersection.width * intersection.height > 0)
					if (abs((detectRects[i].width * detectRects[i].height) - (detectRects[j].width * detectRects[j].height)) < 2000)
						if (intersection.width / intersection.height < 5 &&  intersection.width / intersection.height > 0) {
							Rect lowestYVal;
							int indexHighest;
							if(detectRects[i].y < detectRects[j].y) {
							lowestYVal = detectRects[i]; //higher rectangle
							indexHighest = j;
						} else {
							lowestYVal = detectRects[j]; //higher rectangle
							indexHighest = i;
					}
					if(intersection.y > lowestYVal.y) {
						//cout << "found intersection" << endl;
						detectRects.erase(detectRects.begin()+indexHighest);
					}
				}
			}
		}
	}
}


int main( int argc, const char** argv )
{
	// Flags for various UI features
	bool pause = false;       // pause playback?
	bool printFrames = false; // print frame number?
	int frameDisplayFrequency = 1;

	// Hopefully this turns off any logging
	::google::InitGoogleLogging(argv[0]);

	// Read through command line args, extract
	// cmd line parameters and input filename
	Args args;

	if (!args.processArgs(argc, argv))
		return -2;

	string windowName = "Bin detection"; // GUI window name
	string capPath; // Output directory for captured images
	MediaIn* cap;
	openMedia(args.inputName, cap, capPath, windowName, !args.batchMode);

	GroundTruth groundTruth("ground_truth.txt", args.inputName);
	unsigned groundTruthActual = 0;
	unsigned groundTruthFound  = 0;
	unsigned groundTruthFrame  = 1;
	vector<Rect> groundTruthList;
	vector<unsigned int> groundTruthFrames;
	if (args.groundTruth)
		groundTruthFrames = groundTruth.getFrameList();

	// Seek to start frame if necessary
	if (args.frameStart > 0)
		cap->frameCounter(args.frameStart);

	if (!args.batchMode)
		namedWindow(windowName, WINDOW_AUTOSIZE);


	Mat frame;

	// TODO : Figure this out 
	minDetectSize = cap->width() * 0.05;

	// If UI is up, pop up the parameters window
	if (!args.batchMode)
	{
		string detectWindowName = "Detection Parameters";
		namedWindow(detectWindowName);
		createTrackbar ("Scale", detectWindowName, &scale, 50, NULL);
		createTrackbar ("Neighbors", detectWindowName, &neighbors, 50, NULL);
		createTrackbar ("Min Detect", detectWindowName, &minDetectSize, 200, NULL);
		createTrackbar ("Max Detect", detectWindowName, &maxDetectSize, max(cap->width(), cap->height()), NULL);
	}

	// Create list of tracked objects
	// recycling bins are 24" wide
	TrackedObjectList binTrackingList(24.0, cap->width());

	NetworkTable::SetClientMode();
	NetworkTable::SetIPAddress("10.9.0.2");
	NetworkTable *netTable = NetworkTable::GetTable("VisionTable");
	const size_t netTableArraySize = 7; // 7 bins?
	NumberArray netTableArray;

	// 7 bins max, 3 entries each (confidence, distance, angle)
	netTableArray.setSize(netTableArraySize * 3);

	// Code to write video frames to avi file on disk
	VideoWriter outputVideo;
	VideoWriter markedupVideo;
	args.writeVideo = netTable->GetBoolean("WriteVideo", args.writeVideo);
	const int videoWritePollFrequency = 30; // check for network table entry every this many frames (~5 seconds or so)
	int videoWritePollCount = videoWritePollFrequency;

	FrameTicker frameTicker;

	DetectState detectState(
		  ClassifierIO(args.classifierBaseDir, args.classifierDirNum, args.classifierStageNum),
		  gpu::getCudaEnabledDeviceCount() > 0);

	// Find the first frame number which has ground truth data
	if (args.groundTruth)
		cap->frameCounter(groundTruthFrames[0]);

	// Start of the main loop
	//  -- grab a frame
	//  -- update the angle of tracked objects
	//  -- do a cascade detect on the current frame
	//  -- add those newly detected objects to the list of tracked objects
	while(cap->getNextFrame(frame, pause))
	{
		frameTicker.start(); // start time for this frame
		if (--videoWritePollCount == 0)
		{
			args.writeVideo = netTable->GetBoolean("WriteVideo", args.writeVideo);
			videoWritePollCount = videoWritePollFrequency;
		}

		if (args.writeVideo)
		   writeVideoToFile(outputVideo, getVideoOutName().c_str(), frame, netTable, true);

		//TODO : grab angle delta from robot
		// Adjust the position of all of the detected objects
		// to account for movement of the robot between frames
		double deltaAngle = 0.0;
		binTrackingList.adjustAngle(deltaAngle);

		// This code will load a classifier if none is loaded - this handles
		// initializing the classifier the first time through the loop.
		// It also handles cases where the user changes the classifer
		// being used - this forces a reload
		// Finally, it allows a switch between CPU and GPU on the fly
		if (detectState.update() == false)
			return -1;

		// Apply the classifier to the frame
		// detectRects is a vector of rectangles, one for each detected object
		vector<Rect> detectRects;
		detectState.detector()->Detect(frame, detectRects);
		checkDuplicate(detectRects);

		// If args.captureAll is enabled, write each detected rectangle
		// to their own output image file. Do it before anything else
		// so there's nothing else drawn to frame yet, just the raw
		// input image
		if (args.captureAll)
			for (size_t index = 0; index < detectRects.size(); index++)
				writeImage(frame, detectRects, index, capPath.c_str(), cap->frameCounter());

		// Draw detected rectangles on frame
		if (!args.batchMode && args.rects && ((cap->frameCounter() % frameDisplayFrequency) == 0))
			drawRects(frame,detectRects);

		// Process this detected rectangle - either update the nearest
		// object or add it as a new one
		for(vector<Rect>::const_iterator it = detectRects.begin(); it != detectRects.end(); ++it)
			binTrackingList.processDetect(*it);
		#if 0
		// Print detect status of live objects
		if (args.tracking)
			binTrackingList.print();
		#endif
		// Grab info from trackedobjects. Display it and update network tables
		vector<TrackedObjectDisplay> displayList;
		binTrackingList.getDisplay(displayList);

		// Draw tracking info on display if
		//   a. tracking is toggled on
		//   b. batch (non-GUI) mode isn't active
		//   c. we're on one of the frames to display (every frDispFreq frames)
		if (args.tracking && !args.batchMode && ((cap->frameCounter() % frameDisplayFrequency) == 0))
		    drawTrackingInfo(frame, displayList);

		if (!args.ds)
		{
		   // Clear out network table array
		   for (size_t i = 0; !args.ds & (i < (netTableArraySize * 3)); i++)
			   netTableArray.set(i, -1);

		   for (size_t i = 0; i < min(displayList.size(), netTableArraySize); i++)
		   {
			  netTableArray.set(i*3,   displayList[i].ratio);
			  netTableArray.set(i*3+1, displayList[i].distance);
			  netTableArray.set(i*3+2, displayList[i].angle);
		   }
		   netTable->PutValue("VisionArray", netTableArray);
		}

		// Don't update to next frame if paused to prevent
		// objects missing from this frame to be aged out
		// as the current frame is redisplayed over and over
		if (!pause)
			binTrackingList.nextFrame();

		// For interactive mode, update the FPS as soon as we have
		// a complete array of frame time entries
		// For args.batch mode, only update every frameTicksLength frames to
		// avoid printing too much stuff
	    if (frameTicker.valid() &&
			( (!args.batchMode && ((cap->frameCounter() % frameDisplayFrequency) == 0)) ||
			  ( args.batchMode && ((cap->frameCounter() % 50) == 0))))
	    {
			stringstream ss;
			// If in args.batch mode and reading a video, display
			// the frame count
			int frames = cap->frameCount();
			if (args.batchMode && (frames > 0))
			{
				ss << cap->frameCounter();
				if (frames > 0)
				   ss << '/' << frames;
				ss << " : ";
			}
			// Print the FPS
			ss << fixed << setprecision(2) << frameTicker.getFPS() << "FPS";
			if (!args.batchMode)
				putText(frame, ss.str(), Point(frame.cols - 15 * ss.str().length(), 50), FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,255));
			else
				cout << ss.str() << endl;
	    }

		// Driverstation Code
		if (args.ds)
		{
			// Report boolean value for each bin on the step
			bool hits[4];
			for (int i = 0; i < 4; i++)
			{
				Rect dsRect(i * frame.cols / 4, 0, frame.cols/4, frame.rows);
				if (!args.batchMode && ((cap->frameCounter() % frameDisplayFrequency) == 0))
					rectangle(frame, dsRect, Scalar(0,255,255,3));
				hits[i] = false;
				// For each quadrant of the field, look for a detected
				// rectangle contained entirely in the quadrant
				// Assume that if that's found, it is a bin
				// TODO : Tune this later with a distance range
				for (vector<TrackedObjectDisplay>::const_iterator it = displayList.begin(); it != displayList.end(); ++it)
				{
					if (((it->rect & dsRect) == it->rect) && (it->ratio > 0.15))
					{
						if (!args.batchMode && ((cap->frameCounter() % frameDisplayFrequency) == 0))
							rectangle(frame, it->rect, Scalar(255,128,128), 3);
						hits[i] = true;
					}
				}
				writeNetTableBoolean(netTable, "Bin", i + 1, hits[i]);
			}
		}

		if (cap->frameCount() >= 0)
		{
			groundTruthList    = groundTruth.get(cap->frameCounter());
			groundTruthActual += groundTruthList.size();
			for(vector<Rect>::const_iterator gt = groundTruthList.begin(); gt != groundTruthList.end(); ++gt)
			{
				for(vector<Rect>::const_iterator it = detectRects.begin(); it != detectRects.end(); ++it)
				{
					// If the intersection is > 30% of the area of
					// the ground truth, that's a success
					if ((*it & *gt).area() > (gt->area() * 0.3))
					{
						groundTruthFound += 1;
						if (!args.batchMode && ((cap->frameCounter() % frameDisplayFrequency) == 0))
							rectangle(frame, *it, Scalar(128,128,128), 3);
						break;
					}
				}
			}
		}

		// Various random display updates. Only do them every frameDisplayFrequency
		// frames. Normally this value is 1 so we display every frame. When exporting
		// X over a network, though, we can speed up processing by only displaying every
		// 3, 5 or whatever frames instead.
		if (!args.batchMode && ((cap->frameCounter() % frameDisplayFrequency) == 0))
		{
			// Put an A on the screen if capture-all is enabled so
			// users can keep track of that toggle's mode
			if (args.captureAll)
				putText(frame, "A", Point(25,25), FONT_HERSHEY_PLAIN, 2.5, Scalar(0, 255, 255));

			// Print frame number of video if the option is enabled
			int frames = cap->frameCount();
			if (printFrames && (frames > 0))
			{
				stringstream ss;
				ss << cap->frameCounter() << '/' << frames;
				putText(frame, ss.str(),
				        Point(frame.cols - 15 * ss.str().length(), 20),
						FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,255));
			}

			// Display current classifier under test
			putText(frame, detectState.print(),
			        Point(0, frame.rows - 30), FONT_HERSHEY_PLAIN,
					1.5, Scalar(0,0,255));

			// Display crosshairs so we can line up the camera
			if (args.calibrate)
			{
			   line (frame, Point(frame.cols/2, 0) , Point(frame.cols/2, frame.rows), Scalar(255,255,0));
			   line (frame, Point(0, frame.rows/2) , Point(frame.cols, frame.rows/2), Scalar(255,255,0));
			}

			if (groundTruthList.size())
				drawRects(frame, groundTruthList, Scalar(255,0,0), false);

			// Main call to display output for this frame after all
			// info has been written on it.
			imshow( windowName, frame );

			// If saveVideo is set, write the marked-up frame to a vile
			if (args.saveVideo)
			   writeVideoToFile(markedupVideo, getVideoOutName(false).c_str(), frame, netTable, false);

			char c = waitKey(5);
			if ((c == 'c') || (c == 'q') || (c == 27))
			{ // exit
				if (netTable->IsConnected())
					NetworkTable::Shutdown();
				return 0;
			}
			else if( c == ' ') { pause = !pause; }
			else if( c == 'f')  // advance to next frame
			{
				cap->getNextFrame(frame, false);
			}
			else if (c == 'A') // toggle capture-all
			{
				args.captureAll = !args.captureAll;
			}
			else if (c == 't') // toggle args.tracking info display
			{
				args.tracking = !args.tracking;
			}
			else if (c == 'r') // toggle args.rects info display
			{
				args.rects = !args.rects;
			}
			else if (c == 'a') // save all detected images
			{
				// Save from a copy rather than the original
				// so all the markup isn't saved, only the raw image
				Mat frameCopy;
				cap->getNextFrame(frameCopy, true);
				for (size_t index = 0; index < detectRects.size(); index++)
					writeImage(frameCopy, detectRects, index, capPath.c_str(), cap->frameCounter());
			}
			else if (c == 'p') // print frame number to console
			{
				cout << cap->frameCounter() << endl;
			}
			else if (c == 'P') // Toggle frame # printing to display
			{
				printFrames = !printFrames;
			}
			else if (c == 'S')
			{
				frameDisplayFrequency += 1;
			}
			else if (c == 's')
			{
				frameDisplayFrequency = max(1, frameDisplayFrequency - 1);
			}
			else if (c == 'G') // toggle CPU/GPU mode
			{
				detectState.toggleGPU();
			}
			else if (c == '.') // higher classifier stage
			{
				detectState.changeSubModel(true);
			}
			else if (c == ',') // lower classifier stage
			{
				detectState.changeSubModel(false);
			}
			else if (c == '>') // higher classifier dir num
			{
				detectState.changeModel(true);
			}
			else if (c == '<') // lower classifier dir num
			{
				detectState.changeModel(false);
			}
			else if (isdigit(c)) // save a single detected image
			{
				Mat frameCopy;
				cap->getNextFrame(frameCopy, true);
				writeImage(frameCopy, detectRects, c - '0', capPath.c_str(), cap->frameCounter());
			}
		}

		// Save frame time for the current frame
		frameTicker.end();

		// If testing only ground truth frames, move to the
		// next one in the list
		if (args.groundTruth)
		   cap->frameCounter(groundTruthFrames[groundTruthFrame++]);

		// Skip over frames if needed - useful for batch extracting hard negatives
		// so we don't get negatives from every frame. Sequential frames will be
		// pretty similar so there will be lots of redundant images found
		else if (args.skip > 0)
		   cap->frameCounter(cap->frameCounter() + args.skip - 1);
	}
	if (groundTruthActual)
		cout << groundTruthFound << " of " << groundTruthActual << " ground truth objects found (" << (double)groundTruthFound / groundTruthActual * 100.0 << "%)" << endl;
	return 0;
}

// Write out the selected rectangle from the input frame
// Save multiple copies - the full size image, that full size image converted to grayscale and histogram equalized, and a small version of each.
// The small version is saved because while the input images to the training process are 20x20
// the detection code can find larger versions of them. Scale them down to 20x20 so the complete detected
// image is used as a negative to the training code. Without this, the training code will pull a 20x20
// sub-image out of the larger full image
void writeImage(const Mat &frame, const vector<Rect> &rects, size_t index, const char *path, int frameCounter)
{
   if (index < rects.size())
   {
      Mat image = frame(rects[index]);
      // Create filename, save image
      stringstream fn;
      fn << "negative/" << path << "_" << frameCounter << "_" << index;
      imwrite(fn.str() + ".png", image);

      // Save grayscale equalized version
      Mat frameGray;
      cvtColor( image, frameGray, CV_BGR2GRAY );
      equalizeHist( frameGray, frameGray );
      imwrite(fn.str() + "_g.png", frameGray);

      // Save 20x20 version of the same image
      Mat smallImg;
      resize(image, smallImg, Size(20,20));
      imwrite(fn.str() + "_s.png", smallImg);

      // Save grayscale equalized version of small image
      cvtColor( smallImg, frameGray, CV_BGR2GRAY );
      equalizeHist( frameGray, frameGray );
      imwrite(fn.str() + "_g_s.png", frameGray);
   }
}

string getDateTimeString(void)
{
   time_t rawtime;
   struct tm * timeinfo;

   time (&rawtime);
   timeinfo = localtime (&rawtime);

   stringstream ss;
   ss << timeinfo->tm_mon + 1 << "-" << timeinfo->tm_mday << "_" << timeinfo->tm_hour << "_" << timeinfo->tm_min;
   return ss.str();
}

bool hasSuffix(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
        str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Open video capture object. Figure out if input is camera, video, image, etc
void openMedia(const string &fileName, MediaIn *&cap, string &capPath, string &windowName, bool gui)
{
	// Digit, but no dot (meaning no file extension)? Open camera
	if (fileName.length() == 0 ||
			((fileName.find('.') == string::npos) && isdigit(fileName[0])))
	{
		stringstream ss;
		int camera = fileName.length() ? atoi(fileName.c_str()) : 0;

		cap	= new ZedIn();
		Mat	mat;
		if(!cap->getNextFrame(mat))
		{
			delete cap;
			cap = new C920CameraIn(camera, gui);
			if (!cap->getNextFrame(mat))
			{
				delete cap;
				cap = new CameraIn(camera, gui);
				ss << "Default Camera ";
			}
			else
			{
				ss << "C920 Camera ";
			}
		}
		else
		{
			ss << "Zed Camera ";
		}
		ss << camera;
		windowName = ss.str();
		capPath    = getDateTimeString();
	}
	else // has to be a file name, we hope
	{
		if (hasSuffix(fileName, ".png") || hasSuffix(fileName, ".jpg"))
			cap = new ImageIn(fileName.c_str());
		else
			cap = new VideoIn(fileName.c_str());

		// Strip off directory for capture path
		capPath = fileName;
		const size_t last_slash_idx = capPath.find_last_of("\\/");
		if (std::string::npos != last_slash_idx)
			capPath.erase(0, last_slash_idx + 1);
		windowName = fileName;
	}
}

void writeNetTableBoolean(NetworkTable *netTable, string label, int index, bool value)
{
   stringstream ss;
   ss << label << index+1;
   netTable->PutBoolean(ss.str().c_str(), value);
}

// Video-MM-DD-YY_hr-min-sec-##.avi
string getVideoOutName(bool raw)
{
	int index = 0;
	int rc;
	struct stat statbuf;
	stringstream ss;
	time_t rawtime;
	struct tm * timeinfo;
	time (&rawtime);
	timeinfo = localtime (&rawtime);
	do
	{
		ss.str(string(""));
		ss.clear();
		ss << "Video-" << timeinfo->tm_mon + 1 << "-" << timeinfo->tm_mday << "-" << timeinfo->tm_year+1900 << "_";
		ss << timeinfo->tm_hour << "-" << timeinfo->tm_min << "-" << timeinfo->tm_sec << "-";
		ss << index++;
		if (raw == false)
		   ss << "_processed";
		ss << ".avi";
		rc = stat(ss.str().c_str(), &statbuf);
	}
	while (rc == 0);
	return ss.str();
}

// Write a frame to an output video
// optionally, if dateAndTime is set, stamp the date, time and match information to the frame before writing
void writeVideoToFile(VideoWriter &outputVideo, const char *filename, const Mat &frame, NetworkTable *netTable, bool dateAndTime)
{
   if (!outputVideo.isOpened())
	  outputVideo.open(filename, CV_FOURCC('M','J','P','G'), 15, Size(frame.cols, frame.rows), true);
   WriteOnFrame textWriter(frame);
   if (dateAndTime)
   {
	  string matchNum  = netTable->GetString("Match Number", "No Match Number");
	  double matchTime = netTable->GetNumber("Match Time",-1);
	  textWriter.writeMatchNumTime(matchNum,matchTime);
	  textWriter.writeTime();
   }
   textWriter.write(outputVideo);
}
