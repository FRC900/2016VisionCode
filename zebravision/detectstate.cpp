// This class controls which detector is created
// and used.  It holds the current config files
// for the detector as well as various flags 
// controlling detector state (GPU vs CPU, etc)
//
// Methods are provided to change these settings
// Each frame the code runs update(). If any settings
// have changed since the last frame, the old
// detector is deleted and a new one is created
// with the updated settings.
#include <iostream>
#include <string>

#include "detectstate.hpp"
#ifndef USE_GIE
#include "CaffeClassifier.hpp"
#else
#include "GIEClassifier.hpp"
#endif

using namespace std;
using namespace cv;
using namespace cv::gpu;

// Classifier IO holds the directory which has
// net description, weights, labels, etc.
// It also stores the index of the weight file
// to use - each epoch is saved and we can switch
// between them if needed
// hfov should be removed once the detect call takes
// a tracked object as input
// the flags control which parts of the detector
// use GPU vs CPU code
DetectState::DetectState(const ClassifierIO &d12IO, 
		const ClassifierIO &d24IO,
	   	const ClassifierIO &c12IO, 
		const ClassifierIO &c24IO, 
		float hfov, 
		bool gpuDetect, 
		bool gpuClassifier,
	   	bool gie) :
    detector_(NULL),
	d12IO_(d12IO),
	d24IO_(d24IO),
	c12IO_(c12IO),
	c24IO_(c24IO),
	hfov_(hfov),
	gpuDetect_(gpuDetect),
	gpuClassifier_(gpuClassifier),
	gie_(gie),
	oldGpuDetect_(gpuDetect),
	oldGpuClassifier_(gpuClassifier),
	oldGie_(gie),
	reload_(true)
{
   update();
}

DetectState::~DetectState()
{
	if (detector_)
		delete detector_;
}

// Grab file names needed to load a given classifier
// Check that the results make sense and return
// them to the caller. 
bool DetectState::checkNNetFiles(const ClassifierIO &inCLIO,
								 const string &name,
								 vector<string> &outFiles)
{
    outFiles = inCLIO.getClassifierFiles();
    for (size_t i = 0; i < outFiles.size(); ++i)
    {
        cerr << name << "[" << i << "] = " << outFiles[i] << endl;
    }
    if (outFiles.size() != 4)
    {
        cerr << "Wrong number of " << name << " to load classifier" << endl;
        return false;
    }
	return true;
}

// Called each frame. Reloads the detector if
// any settings have changed
bool DetectState::update(void)
{
	if (reload_ == false)
		return true;

	vector<string> d12Files;
	vector<string> d24Files;
	vector<string> c12Files;
	vector<string> c24Files;

	if (!checkNNetFiles(d12IO_, "D12Files", d12Files) ||
		!checkNNetFiles(d24IO_, "C24Files", d24Files) ||
		!checkNNetFiles(c12IO_, "D12Files", c12Files) ||
		!checkNNetFiles(c24IO_, "C24Files", c24Files))
		return false;

	// Save old detector state in case a problem
	// occurs
	ObjDetect *oldDetector = detector_;

	// Decision tree on which detector and classifier
	// to run.  Some of these combinations might not make
	// sense to maybe prune them down after some testing?
#ifndef USE_GIE
	//if (!gie_)
	{
		if (!gpuClassifier_)
		{
			if (!gpuDetect_)
				detector_ = new ObjDetectCPUCaffeCPU(d12Files, d24Files, c12Files, c24Files, hfov_);
			else
				detector_ = new ObjDetectGPUCaffeCPU(d12Files, d24Files, c12Files, c24Files, hfov_);
		}
		else
		{
			if (!gpuDetect_)
				detector_ = new ObjDetectCPUCaffeGPU(d12Files, d24Files, c12Files, c24Files, hfov_);
			else
				detector_ = new ObjDetectGPUCaffeGPU(d12Files, d24Files, c12Files, c24Files, hfov_);
		}
	}
#else
	//else
	{
		// GIE implies GPU detection - CPU doesn't make sense there
		if (!gpuClassifier_)
			detector_ = new ObjDetectCPUGIEGPU(d12Files, d24Files, c12Files, c24Files, hfov_);
		else
			detector_ = new ObjDetectGPUGIEGPU(d12Files, d24Files, c12Files, c24Files, hfov_);
	}
#endif

	reload_ = false;

	// Verfiy the load
	if( !detector_ || !detector_->initialized() )
	{
		cerr << "Error loading detector" << endl;
		detector_ = oldDetector;
		gpuDetect_ = oldGpuDetect_;
		gpuClassifier_ = oldGpuClassifier_;
		gie_ = oldGie_;
		return (oldDetector != NULL);
	}

	if (oldDetector)
		delete oldDetector;
	oldGpuDetect_ = gpuDetect_;
	oldGpuClassifier_ = gpuClassifier_;
	oldGie_ = gie_;

	return true;
}

void DetectState::toggleGPUDetect(void)
{
	if (getCudaEnabledDeviceCount() > 0)
	{
		gpuDetect_ = !gpuDetect_;
		reload_ = true;
	}
}

void DetectState::toggleGPUClassifier(void)
{
	if (getCudaEnabledDeviceCount() > 0)
	{
		gpuClassifier_ = !gpuClassifier_;
		reload_ = true;
	}
}
void DetectState::toggleGIE(void)
{
	if (getCudaEnabledDeviceCount() > 0)
	{
		gie_ = !gie_;
		reload_ = true;
	}
}

void DetectState::changeD12SubModel(bool increment)
{
   if (d12IO_.findNextClassifierStage(increment))
	  reload_ = true;
}

void DetectState::changeD12Model(bool increment)
{
   if (d12IO_.findNextClassifierDir(increment))
	  reload_ = true;
}

void DetectState::changeD24SubModel(bool increment)
{
   if (d24IO_.findNextClassifierStage(increment))
	  reload_ = true;
}

void DetectState::changeD24Model(bool increment)
{
   if (d24IO_.findNextClassifierDir(increment))
	  reload_ = true;
}

void DetectState::changeC12SubModel(bool increment)
{
   if (c12IO_.findNextClassifierStage(increment))
	  reload_ = true;
}

void DetectState::changeC12Model(bool increment)
{
   if (c12IO_.findNextClassifierDir(increment))
	  reload_ = true;
}

void DetectState::changeC24SubModel(bool increment)
{
   if (c24IO_.findNextClassifierStage(increment))
	  reload_ = true;
}

void DetectState::changeC24Model(bool increment)
{
	if (c24IO_.findNextClassifierDir(increment))
		reload_ = true;
}

std::string DetectState::print(void) const
{
	string ret;
	if (gpuDetect_)
		ret += "G_";
	else
		ret += "C_";
	if (gpuClassifier_)
		ret += "G_";
	else
		ret += "C_";
	if (gie_)
		ret += "GIE";
	else
		ret += "Caffe";
	ret += " " + d12IO_.print() + "," + d24IO_.print() + "," + c12IO_.print() + "," + c24IO_.print();
	return ret;
}
