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

#include <zmq.hpp>

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
#include "track3d.hpp"
#include "Args.hpp"
#include "WriteOnFrame.hpp"
#include "GoalDetector.hpp"
#include "FovisLocalizer.hpp"
#include "Utilities.hpp"

using namespace std;
using namespace cv;
using namespace utils;

const float HFOV = 84.14 * (M_PI / 180.0);  
const float VFOV = 53.836 * (M_PI / 180.0); 

//function prototypes
void writeImage(const Mat &frame, const vector<Rect> &rects, size_t index, const char *path, int frameCounter);
string getDateTimeString(void);
void drawRects(Mat image ,vector<Rect> detectRects, Scalar rectColor = Scalar(0,0,255), bool text = true);
void drawTrackingInfo(Mat &frame, vector<TrackedObjectDisplay> &displayList);
void openMedia(const string &fileName, MediaIn *&cap, string &capPath, string &windowName, bool gui);
void openVideoCap(const string &fileName, VideoIn *&cap, string &capPath, string &windowName, bool gui);
string getVideoOutName(bool raw = true);
void writeVideoToFile(VideoWriter &outputVideo, const char *filename, const Mat &frame, void *netTable, bool dateAndTime);

void drawRects(Mat image, vector<Rect> detectRects, Scalar rectColor, bool text)
{
    for(auto it = detectRects.cbegin(); it != detectRects.cend(); ++it)
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
   for (auto it = displayList.cbegin(); it != displayList.cend(); ++it)
   {
	  if (it->ratio >= 0.05)
	  {
		 const int roundPosTo = 2;
		 std::cout << "Ratio: " << it->ratio << std::endl;
		 // Color moves from red to green (via brown, yuck)
		 // as the detected ratio goes up
		 Scalar rectColor(0, 255 * it->ratio, 255 * (1.0 - it->ratio));
		 // Highlight detected target
		 rectangle(frame, it->rect, rectColor, 3);
		 // Write detect ID, distance and angle data
		 putText(frame, it->id, Point(it->rect.x+25, it->rect.y+30), FONT_HERSHEY_PLAIN, 2.0, rectColor);
		 stringstream label;
		 label << fixed << setprecision(roundPosTo);
		 label << "(" << it->position.x << "," << it->position.y << "," << it->position.z << ")";
		 putText(frame, label.str(), Point(it->rect.x+10, it->rect.y-10), FONT_HERSHEY_PLAIN, 1.2, rectColor);
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

	string windowName = "Ball Detection"; // GUI window name
	string capPath; // Output directory for captured images
	MediaIn* cap;
	openMedia(args.inputName, cap, capPath, windowName, !args.batchMode);

	GroundTruth groundTruth("ground_truth.txt", args.inputName);
	vector<Rect> groundTruthList;

	// Seek to start frame if necessary
	if (args.frameStart > 0)
		cap->frameCounter(args.frameStart);

	if (!args.batchMode)
		namedWindow(windowName, WINDOW_AUTOSIZE);

	Mat frame;

	// TODO : Figure this out 
	//minDetectSize = cap->width() * 0.05;
	minDetectSize = 40;

	// If UI is up, pop up the parameters window
	if (!args.batchMode)
	{
		string detectWindowName = "Detection Parameters";
		namedWindow(detectWindowName);
		createTrackbar ("Scale", detectWindowName, &scale, 50);
		createTrackbar ("NMS Threshold", detectWindowName, &nmsThreshold, 100);
		createTrackbar ("Min Detect", detectWindowName, &minDetectSize, 200);
		createTrackbar ("Max Detect", detectWindowName, &maxDetectSize, max(cap->width(), cap->height()));
		createTrackbar ("D12 Threshold", detectWindowName, &d12Threshold, 100);
		createTrackbar ("D24 Threshold", detectWindowName, &d24Threshold, 100);
	}

	// Create list of tracked objects
	// balls / boulders are 8" wide?
	TrackedObjectList objectTrackingList(Point(cap->width(),cap->height()), Point2f(HFOV,VFOV));
	
	zmq::context_t context (1);
	zmq::socket_t publisher(context, ZMQ_PUB);

	std::cout<< "Starting network publisher 5800" << std::endl;
	publisher.bind("tcp://*:5800");

	const size_t netTableArraySize = 7; // 7 bins?

	// Code to write video frames to avi file on disk
	VideoWriter outputVideo;
	VideoWriter markedupVideo;
	const int videoWritePollFrequency = 30; // check for network table entry every this many frames (~5 seconds or so)
	int videoWritePollCount = videoWritePollFrequency;

	FrameTicker frameTicker;

	DetectState detectState(
		  ClassifierIO(args.classifierBaseDir, args.classifierDirNum, args.classifierStageNum),
		  gpu::getCudaEnabledDeviceCount() > 0);

	// Find the first frame number which has ground truth data
	if (args.groundTruth)
	{
		int frameNum = groundTruth.nextFrameNumber();
		if (frameNum == -1)
			return 0;
		cap->frameCounter(frameNum);
	}

	cap->getNextFrame(frame, pause);
	FovisLocalizer fvlc(cap->getCameraParams(true), cap->width(), cap->height(), frame);
	//Creating Goaldetection object

	GoalDetector gd;
  	Mat depth;
	Rect boundRect;

	int64 stepTimer;	
	
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
			//args.writeVideo = netTable->GetBoolean("WriteVideo", args.writeVideo);
			videoWritePollCount = videoWritePollFrequency;
		}
		
		if (args.writeVideo)
		{
		   writeVideoToFile(outputVideo, getVideoOutName().c_str(), frame, NULL, true);
		}

		

		// This code will load a classifier if none is loaded - this handles
		// initializing the classifier the first time through the loop.
		// It also handles cases where the user changes the classifer
		// being used - this forces a reload
		// Finally, it allows a switch between CPU and GPU on the fly
		if (detectState.update() == false)
		break;

		//Getting depth matrix
		cap->getDepthMat(depth);

		//run Goaldetector and FovisLocator code

		stepTimer = cv::getTickCount();
		gd.processFrame(frame, depth, boundRect);
		cout << "Time to run goaldetection - " << ((double)cv::getTickCount() - stepTimer) / getTickFrequency() << endl;

		float gdistance = gd.dist_to_goal();
		float gangle = gd.angle_to_goal();

		stepTimer = cv::getTickCount();
		fvlc.processFrame(frame,depth);
		cout << "Time to fovis - " << ((double)cv::getTickCount() - stepTimer) / getTickFrequency() << endl;

		// Apply the classifier to the frame
		// detectRects is a vector of rectangles, one for each detected object
		vector<Rect> detectRects;
		Mat dummyMat;

		stepTimer = cv::getTickCount();
		detectState.detector()->Detect(frame, dummyMat, detectRects);
		cout << "Time to detect - " << ((double)cv::getTickCount() - stepTimer) / getTickFrequency() << endl;

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

		//adjust locations of objects based on fovis results

		objectTrackingList.adjustLocation(fvlc.transform_eigen());

		// Process this detected rectangle - either update the nearest
		// object or add it as a new one
		//also compute the average depth of the region since that is necessary for the processing
		float depthRectScale = 0.2;
		for(auto it = detectRects.cbegin(); it != detectRects.cend(); ++it) {
			Rect depthRect = *it;
			
			shrinkRect(depthRect,depthRectScale);
			Mat emptyMask(depth.rows,depth.cols,CV_8UC1,Scalar(255));
			
			objectTrackingList.processDetect(*it, minOfDepthMat(depth, emptyMask, depthRect, 10).first, ObjectType(1));
			cout << "Object depth: " << minOfDepthMat(depth, emptyMask, depthRect, 10).first << endl;
		}

		// Grab info from trackedobjects. Display it and update zmq subscribers
		vector<TrackedObjectDisplay> displayList;
		objectTrackingList.getDisplay(displayList);
		//Creates immutable strings for 0MQ Output
		stringstream zmqString;
		zmqString << "V ";
		stringstream gString;
		gString << "G ";
		gString << fixed << setprecision(2) << gdistance << " ";
		gString << fixed << setprecision(2) << gangle;
		// Draw tracking info on display if
		//   a. tracking is toggled on
		//   b. batch (non-GUI) mode isn't active
		//   c. we're on one of the frames to display (every frameDispFreq frames)
		if (args.tracking && !args.batchMode && ((cap->frameCounter() % frameDisplayFrequency) == 0)) {
		    drawTrackingInfo(frame, displayList);

			//create a top view image of the robot and all detected objects
			Range top_xRange = Range(-2,2);
			Range top_yRange = Range(-2,2);
			Point top_imageSize = Point(640,640);
			Point top_imageCenter = Point(top_imageSize.x / 2, top_imageSize.y / 2);
			int top_rectSize = 40;

			Mat top_frame(top_imageSize.y,top_imageSize.x,CV_8UC3, Scalar(0,0,0) );
			circle(top_frame,top_imageCenter, 10, Scalar(0,0,255));
			line(top_frame, top_imageCenter, top_imageCenter - Point(0,top_imageSize.x / 2), Scalar(0,0,255), 3);
			for (auto it = displayList.cbegin(); it != displayList.cend(); ++it)
		   	{
				Point2f top_realPos = Point2f(it->position.x, it->position.y);
			  	Point top_imagePos;
				top_imagePos.x = top_realPos.x * (top_imageSize.x / top_xRange.size()) + (top_imageSize.x / 2);
				top_imagePos.y = -(top_realPos.y * (top_imageSize.y / top_yRange.size())) + (top_imageSize.y / 2);
				circle(top_frame, top_imagePos, top_rectSize, Scalar(255,0,0), 5);
				
		   	}
			imshow("Top view", top_frame);
		}

		for (size_t i = 0; i < netTableArraySize; i++)
		{
			if (i < displayList.size())
			{
				zmqString << fixed << setprecision(2) << displayList[i].ratio << " " ;
				zmqString << fixed << setprecision(2) << displayList[i].position.x << " " ;
				zmqString << fixed << setprecision(2) << displayList[i].position.y << " " ;
				zmqString << fixed << setprecision(2) << displayList[i].position.z << " " ;
			}
			else
				zmqString << "0.00 0.00 0.00 ";
		}

		//cout << "ZMQ : " << zmqString.str().length() <<  " : " << zmqString.str() << endl;
		//cout << "G : " << gString.str().length() << " : " << gString.str() << endl;
		zmq::message_t request(zmqString.str().length() - 1);
		zmq::message_t grequest(gString.str().length() - 1);
		memcpy((void *)request.data(), zmqString.str().c_str(), zmqString.str().length() - 1);
		memcpy((void *)grequest.data(), gString.str().c_str(), gString.str().length() - 1);
		publisher.send(request);
		publisher.send(grequest);

		// Don't update to next frame if paused to prevent
		// objects missing from this frame to be aged out
		// as the current frame is redisplayed over and over
		if (!pause)
			objectTrackingList.nextFrame();

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
			//else
				//cout << ss.str() << endl;
	    }

		// Check ground truth data on videos and images,
		// but not on camera input
		vector<Rect> groundTruthHitList;
		if (cap->frameCount() >= 0)
			groundTruthHitList = groundTruth.processFrame(cap->frameCounter() - 1, detectRects);


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

			// Draw ground truth info for this frame. Will be a no-op
			// if none is available for this particular video frame
			drawRects(frame, groundTruth.get(cap->frameCounter() - 1), Scalar(255,0,0), false);
			drawRects(frame, groundTruthHitList, Scalar(128, 128, 128), false);
			rectangle(frame, boundRect, Scalar(255,0,0));

			// Main call to display output for this frame after all
			// info has been written on it.
			imshow(windowName, frame);

			// If saveVideo is set, write the marked-up frame to a vile
			if (args.saveVideo)
			   writeVideoToFile(markedupVideo, getVideoOutName(false).c_str(), frame, NULL, false);

			// Process user input for this frame
			char c = waitKey(5);
			if ((c == 'c') || (c == 'q') || (c == 27))
			{ // exit
				break;
			}
			else if( c == ' ')  // Toggle pause
			{ 
				pause = !pause; 
			}
			else if( c == 'f')  // advance to next frame
			{
				if (!pause)
					pause = true;
				if (args.groundTruth)
				{
					int frame = groundTruth.nextFrameNumber();
					// Exit if no more frames left to test
					if (frame == -1)
						break;
					// Otherwise, if not paused, move to the next frame
					cap->frameCounter(frame);
				}
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

		// If testing only ground truth frames, move to the
		// next one in the list
		if (args.groundTruth)
		{
			int frame = groundTruth.nextFrameNumber();
			// Exit if no more frames left to test
			if (frame == -1)
				break;
			// Otherwise, if not paused, move to the next frame
			if (!pause)
				cap->frameCounter(frame);
		}
		// Skip over frames if needed - useful for batch extracting hard negatives
		// so we don't get negatives from every frame. Sequential frames will be
		// pretty similar so there will be lots of redundant images found
		else if (!pause && (args.skip > 0))
		{	
			// Exit if the next skip puts the frame beyond the end of the video
			if ((cap->frameCounter() + args.skip) >= cap->frameCount())
				break;
			cap->frameCounter(cap->frameCounter() + args.skip - 1);
		}

		// Check for running still images in batch mode - only
		// process the image once rather than looping forever
		if (args.batchMode && (cap->frameCount() == 1))
			break;
	
		// Save frame time for the current frame
		frameTicker.end();
	}
	groundTruth.print();

	return 0;
}

// Write out the selected rectangle from the input frame
void writeImage(const Mat &frame, const vector<Rect> &rects, size_t index, const char *path, int frameCounter)
{
   mkdir("negative", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
   if (index < rects.size())
   {
      // Create filename, save image
      stringstream fn;
      fn << "negative/" << path << "_" << frameCounter << "_" << index;
      imwrite(fn.str() + ".png", frame(rects[index]));
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
		if ((hasSuffix(fileName, ".png") || hasSuffix(fileName, ".jpg") ||
		     hasSuffix(fileName, ".PNG") || hasSuffix(fileName, ".JPG")))
			cap = new ImageIn(fileName.c_str());
		else if (hasSuffix(fileName, ".svo"))
			cap = new ZedIn(fileName.c_str());
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
void writeVideoToFile(VideoWriter &outputVideo, const char *filename, const Mat &frame, void *netTable, bool dateAndTime)
{
   if (!outputVideo.isOpened())
	   outputVideo.open(filename, CV_FOURCC('M','J','P','G'), 15, Size(frame.cols, frame.rows), true);
   WriteOnFrame textWriter(frame);
   if (dateAndTime)
   {
	   (void)netTable;
	   //string matchNum  = netTable->GetString("Match Number", "No Match Number");
	   //double matchTime = netTable->GetNumber("Match Time",-1);
	   string matchNum = "No Match Number";
	   double matchTime = -1;
	   textWriter.writeMatchNumTime(matchNum,matchTime);
	   textWriter.writeTime();
   }
   textWriter.write(outputVideo);
}
