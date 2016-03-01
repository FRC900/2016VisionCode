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

#include "detectstate.hpp"
#include "frameticker.hpp"
#include "groundtruth.hpp"
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

static const float HFOV = 51.3 * (M_PI / 180.0);  
static const float VFOV = HFOV * 480./ 640.; // pixels are square

//function prototypes
void writeImage(const Mat &frame, const vector<Rect> &rects, size_t index, const char *path, int frameNumber);
string getDateTimeString(void);
void drawRects(Mat image ,vector<Rect> detectRects, Scalar rectColor = Scalar(0,0,255), bool text = true);
void drawTrackingInfo(Mat &frame, vector<TrackedObjectDisplay> &displayList);
void drawTrackingTopDown(Mat &frame, vector<TrackedObjectDisplay> &displayList);
void openMedia(const string &fileName, MediaIn *&cap, string &capPath, string &windowName, bool gui, bool &writeVideo);
void openVideoCap(const string &fileName, VideoIn *&cap, string &capPath, string &windowName, bool gui);
string getVideoOutName(bool raw = true, bool zms = false);
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

void drawTrackingTopDown(Mat &frame, vector<TrackedObjectDisplay> &displayList)
{
	//create a top view image of the robot and all detected objects
	Range xRange = Range(-4,4);
	Range yRange = Range(-4,4);
	Point imageSize = Point(640,640);
	Point imageCenter = Point(imageSize.x / 2, imageSize.y / 2);
	int rectSize = 40;

	frame = Mat(imageSize.y,imageSize.x,CV_8UC3, Scalar(0,0,0) );
	circle(frame,imageCenter, 10, Scalar(0,0,255));
	line(frame, imageCenter, imageCenter - Point(0,imageSize.x / 2), Scalar(0,0,255), 3);
	for (auto it = displayList.cbegin(); it != displayList.cend(); ++it)
	{
		Point2f realPos = Point2f(it->position.x, it->position.y);
		Point2f imagePos;
		imagePos.x = realPos.x * (imageSize.x / xRange.size()) + (imageSize.x / 2.0);
		imagePos.y = -(realPos.y * (imageSize.y / yRange.size())) + (imageSize.y / 2.0);
		circle(frame, imagePos, rectSize, Scalar(255,0,0), 5);

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
	openMedia(args.inputName, cap, capPath, windowName, 
			  !args.batchMode, args.writeVideo);

	GroundTruth groundTruth("ground_truth.txt", args.inputName);
	vector<Rect> groundTruthList;

	// Seek to start frame if necessary
	if (args.frameStart > 0)
		cap->frameNumber(args.frameStart);

	if (!args.batchMode)
		namedWindow(windowName, WINDOW_AUTOSIZE);

	// Current frame data - BGR image and depth data (if available)
	Mat frame;
  	Mat depth;
	Mat top_frame; // top-down view of tracked objects

	// TODO : Figure this out 
	//minDetectSize = cap->width() * 0.05;
	minDetectSize = 40;

	// If UI is up, pop up the parameters window
	if (!args.batchMode)
	{
		string detectWindowName = "Detection Parameters";
		namedWindow(detectWindowName);
		createTrackbar ("Scale", detectWindowName, &scale, 50);
		createTrackbar ("D12 NMS Threshold", detectWindowName, &d12NmsThreshold, 100);
		createTrackbar ("D24 NMS Threshold", detectWindowName, &d24NmsThreshold, 100);
		createTrackbar ("Min Detect", detectWindowName, &minDetectSize, 200);
		createTrackbar ("Max Detect", detectWindowName, &maxDetectSize, max(cap->width(), cap->height()));
		createTrackbar ("D12 Threshold", detectWindowName, &d12Threshold, 100);
		createTrackbar ("D24 Threshold", detectWindowName, &d24Threshold, 100);
	}

	// Create list of tracked objects
	// balls / boulders are 8" wide?
	TrackedObjectList objectTrackingList(Size(cap->width(),cap->height()), Point2f(HFOV,VFOV));
	
	zmq::context_t context(1);
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
		  ClassifierIO(args.d12BaseDir, args.d12DirNum, args.d12StageNum),
		  ClassifierIO(args.d24BaseDir, args.d24DirNum, args.d24StageNum),
		  gpu::getCudaEnabledDeviceCount() > 0);

	// Find the first frame number which has ground truth data
	if (args.groundTruth)
	{
		int frameNum = groundTruth.nextFrameNumber();
		if (frameNum == -1)
			return 0;
		cap->frameNumber(frameNum);
	}

	cap->getNextFrame(frame, pause);
	FovisLocalizer fvlc(cap->getCameraParams(true), frame);

	//Creating Goaldetection object
	GoalDetector gd(Point2f(HFOV,VFOV), Size(cap->width(),cap->height()));

	int64 stepTimer;	
	
	// Start of the main loop
	//  -- grab a frame
	//  -- update the angle of tracked objects
	//  -- do a cascade detect on the current frame
	//  -- add those newly detected objects to the list of tracked objects
	while(cap->getNextFrame(frame, pause))
	{
		frameTicker.mark(); // mark start of new frame

		//Getting depth matrix
		cap->getDepthMat(depth);

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

		//run Goaldetector and FovisLocator code
		Rect goalBoundRect;
		gd.processFrame(frame, depth, goalBoundRect);

		float gDistance = gd.dist_to_goal();
		cout << "distance to goal: " << gDistance;
		float gAngle = gd.angle_to_goal();
		cout << " angle to goal: " << gAngle << endl;

		//stepTimer = cv::getTickCount();
		fvlc.processFrame(frame,depth);
		//cout << "Time to fovis - " << ((double)cv::getTickCount() - stepTimer) / getTickFrequency() << endl;

		// Apply the classifier to the frame
		// detectRects is a vector of rectangles, one for each detected object
		stepTimer = cv::getTickCount();
		vector<Rect> detectRects;
		detectState.detector()->Detect(frame, depth, detectRects);
		cout << "Time to detect - " << ((double)cv::getTickCount() - stepTimer) / getTickFrequency() << endl;

		// If args.captureAll is enabled, write each detected rectangle
		// to their own output image file. Do it before anything else
		// so there's nothing else drawn to frame yet, just the raw
		// input image
		if (args.captureAll)
			for (size_t index = 0; index < detectRects.size(); index++)
				writeImage(frame, detectRects, index, capPath.c_str(), cap->frameNumber());

		// Draw detected rectangles on frame
		if (!args.batchMode && args.rects && ((cap->frameNumber() % frameDisplayFrequency) == 0))
			drawRects(frame,detectRects);

		//adjust locations of objects based on fovis results
		utils::printIsometry(fvlc.transform_eigen());

		cout << "Locations before adjustment: " << endl;
		objectTrackingList.print();

		objectTrackingList.adjustLocation(fvlc.transform_eigen());

		cout << "Locations after adjustment: " << endl;
		objectTrackingList.print();

		// Process detected rectangles - either match up with the nearest object
		// add it as a new one
		// also compute the average depth of the region since that is necessary for the processing
		stepTimer = cv::getTickCount();
		vector<Rect>depthFilteredDetectRects;
		vector<float> depths;
		vector<ObjectType> objTypes;
		const float depthRectScale = 0.2;
		for(auto it = detectRects.cbegin(); it != detectRects.cend(); ++it) 
		{
			cout << "Detected object at: " << *it;
			Rect depthRect = *it;
			
			shrinkRect(depthRect,depthRectScale);
			Mat emptyMask(depth.rows,depth.cols,CV_8UC1,Scalar(255));
			float objectDepth = minOfDepthMat(depth, emptyMask, depthRect, 10).first;
			cout << " Depth: " << objectDepth << endl;
			if(objectDepth > 0)
			{
				depthFilteredDetectRects.push_back(*it);
				depths.push_back(objectDepth);
				objTypes.push_back(ObjectType(1));
			}
		} 
		objectTrackingList.processDetect(depthFilteredDetectRects, depths, objTypes);
		cout << "Time to process detect - " << ((double)cv::getTickCount() - stepTimer) / getTickFrequency() << endl;

		// Grab info from trackedobjects. Display it and update zmq subscribers
		vector<TrackedObjectDisplay> displayList;
		objectTrackingList.getDisplay(displayList);

		//Creates immutable strings for 0MQ Output
		stringstream gString;
		gString << "G ";
		gString << fixed << setprecision(2) << gDistance << " ";
		gString << fixed << setprecision(2) << gAngle;

		// Draw tracking info on display if
		//   a. tracking is toggled on
		//   b. batch (non-GUI) mode isn't active
		//   c. we're on one of the frames to display (every frameDispFreq frames)
		if (args.tracking && 
			!args.batchMode && 
			((cap->frameNumber() % frameDisplayFrequency) == 0))
		{
		    drawTrackingInfo(frame, displayList);

			drawTrackingTopDown(top_frame, displayList);
			imshow("Top view", top_frame);
		}

		stringstream zmqString;
		zmqString << "B ";
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
				zmqString << "0.00 0.00 0.00 0.00 ";
		}

		cout << "B : " << zmqString.str().length() <<  " : " << zmqString.str() << endl;
		cout << "G : " << gString.str().length() << " : " << gString.str() << endl;
		zmq::message_t request(zmqString.str().length() - 1);
		zmq::message_t grequest(gString.str().length() - 1);
		memcpy((void *)request.data(), zmqString.str().c_str(), zmqString.str().length() - 1);
		memcpy((void *)grequest.data(), gString.str().c_str(), gString.str().length() - 1);
		publisher.send(request);
		publisher.send(grequest);

		// For interactive mode, update the FPS as soon as we have
		// a complete array of frame time entries
		// For args.batch mode, only update every frameTicksLength frames to
		// avoid printing too much stuff
	    if (frameTicker.valid() &&
			( (!args.batchMode && ((cap->frameNumber() % frameDisplayFrequency) == 0)) ||
			  ( args.batchMode && (((cap->frameNumber() * (args.skip > 0) ? args.skip : 1) % 50) == 0))))
	    {
			stringstream ss;
			// If in args.batch mode and reading a video, display
			// the frame count
			int frames = cap->frameCount();
			if (args.batchMode)
			{
				ss << cap->frameNumber();
				if (frames > 0)
				   ss << '/' << frames;
				ss << " : ";
			}
			// Print the FPS
			ss << fixed << setprecision(2) << frameTicker.getFPS() << "FPS";
			if (!args.batchMode)
				putText(frame, ss.str(), Point(frame.cols - 15 * ss.str().length(), 50), FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,255));
			else
				cerr << ss.str() << endl;
	    }

		// Check ground truth data on videos and images,
		// but not on camera input
		vector<Rect> groundTruthHitList;
		if (cap->frameCount() >= 0)
			groundTruthHitList = groundTruth.processFrame(cap->frameNumber() - 1, detectRects);


		// Various random display updates. Only do them every frameDisplayFrequency
		// frames. Normally this value is 1 so we display every frame. When exporting
		// X over a network, though, we can speed up processing by only displaying every
		// 3, 5 or whatever frames instead.
		if (!args.batchMode && ((cap->frameNumber() % frameDisplayFrequency) == 0))
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
				ss << cap->frameNumber() << '/' << frames;
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
			drawRects(frame, groundTruth.get(cap->frameNumber() - 1), Scalar(255,0,0), false);
			drawRects(frame, groundTruthHitList, Scalar(128, 128, 128), false);

			rectangle(frame, goalBoundRect, Scalar(255,0,0));

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
					cap->frameNumber(frame);
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
					writeImage(frameCopy, detectRects, index, capPath.c_str(), cap->frameNumber());
			}
			else if (c == 'p') // print frame number to console
			{
				cout << cap->frameNumber() << endl;
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
				detectState.changeD12SubModel(true);
			}
			else if (c == ',') // lower classifier stage
			{
				detectState.changeD12SubModel(false);
			}
			else if (c == '>') // higher classifier dir num
			{
				detectState.changeD12Model(true);
			}
			else if (c == '<') // lower classifier dir num
			{
				detectState.changeD12Model(false);
			}
			else if (c == 'm') // higher classifier stage
			{
				detectState.changeD24SubModel(true);
			}
			else if (c == 'n') // lower classifier stage
			{
				detectState.changeD24SubModel(false);
			}
			else if (c == 'M') // higher classifier dir num
			{
				detectState.changeD24Model(true);
			}
			else if (c == 'N') // lower classifier dir num
			{
				detectState.changeD24Model(false);
			}
			else if (isdigit(c)) // save a single detected image
			{
				Mat frameCopy;
				cap->getNextFrame(frameCopy, true);
				writeImage(frameCopy, detectRects, c - '0', capPath.c_str(), cap->frameNumber());
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
				cap->frameNumber(frame);
		}
		// Skip over frames if needed - useful for batch extracting hard negatives
		// so we don't get negatives from every frame. Sequential frames will be
		// pretty similar so there will be lots of redundant images found
		else if (!pause && (args.skip > 0))
		{	
			// Exit if the next skip puts the frame beyond the end of the video
			if ((cap->frameNumber() + args.skip) >= cap->frameCount())
				break;
			cap->frameNumber(cap->frameNumber() + args.skip - 1);
		}

		// Check for running still images in batch mode - only
		// process the image once rather than looping forever
		if (args.batchMode && (cap->frameCount() == 1))
			break;
	}
	groundTruth.print();

	return 0;
}

// Write out the selected rectangle from the input frame
void writeImage(const Mat &frame, const vector<Rect> &rects, size_t index, const char *path, int frameNumber)
{
   mkdir("negative", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
   if (index < rects.size())
   {
      // Create filename, save image
      stringstream fn;
      fn << "negative/" << path << "_" << frameNumber << "_" << index;
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
void openMedia(const string &fileName, MediaIn *&cap, string &capPath, string &windowName, bool gui, bool &writeVideo)
{
	// Digit, but no dot (meaning no file extension)? Open camera
	if (fileName.length() == 0 ||
		((fileName.find('.') == string::npos) && isdigit(fileName[0])))
	{
		stringstream ss;
		int camera = fileName.length() ? atoi(fileName.c_str()) : 0;

		cap = new ZedIn(NULL, writeVideo ? getVideoOutName(true, true).c_str() : NULL );
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
			writeVideo = false;
		}
		ss << camera;
		windowName = ss.str();
		capPath    = getDateTimeString();
	}
	else // has to be a file name, we hope
	{
		if (hasSuffix(fileName, ".png") || hasSuffix(fileName, ".jpg") ||
		    hasSuffix(fileName, ".PNG") || hasSuffix(fileName, ".JPG"))
			cap = new ImageIn(fileName.c_str());
		else if (hasSuffix(fileName, ".svo") || hasSuffix(fileName, ".SVO") ||
		         hasSuffix(fileName, ".zms") || hasSuffix(fileName, ".ZMS"))
		{
			cap = new ZedIn(fileName.c_str(), writeVideo ? getVideoOutName(true, true).c_str() : NULL);
			writeVideo = false;
		}
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
string getVideoOutName(bool raw, bool zms)
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
		if (zms == false)
			ss << ".avi";
		else
			ss << ".zms";
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
