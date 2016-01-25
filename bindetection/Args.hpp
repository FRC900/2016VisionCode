#ifndef INC__ARGS_HPP__
#define INC__ARGS_HPP__

#include <string>

class Args //class for processing arguments
{
	public :
		bool captureAll;       // capture all found targets to image files?
		bool tracking;         // display tracking info?
		bool rects;            // display frame by frame hit info
		bool batchMode;        // non-interactive mode - no display, run through
						       // as quickly as possible. Combine with --all?
		bool ds;               // driver-station?
		int  skip;             // skip over frames (video file input only)
		bool writeVideo;       // write captured video to output
		bool saveVideo;        // write processed video to output
		int  frameStart;       // frame number to start from
		bool calibrate;        // crosshair to calibrate camera
		std::string classifierBaseDir; // base directory for classifier
		int  classifierDirNum; // classifier directory and 
		int  classifierStageNum;// stage to use
		std::string inputName; // input file name or camera number
		bool groundTruth;      // only test frames with ground truth data

		Args(void);
		bool processArgs(int argc, const char **argv);
};

#endif
