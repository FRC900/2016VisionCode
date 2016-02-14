#include <iostream>
#include <string>

#include "detectstate.hpp"

using namespace std;

DetectState::DetectState(const ClassifierIO &classifierIO, bool gpu) :
    detector_(NULL),
	classifierIO_(classifierIO),
	gpu_(gpu),
	reload_(true)
{
   update();
}

bool DetectState::update(void)
{
   if (reload_ == false)
	  return true;

	//string name = classifierIO_.getClassifierName();
	//cerr << name << endl;
	if (detector_)
	   delete detector_;

	// Create a new CPU or GPU classifier based on the
	// user's selection
	//if (gpu_)
	//	detector_ = new GPU_CascadeDetect(name.c_str());
	//else
    vector<string> files = classifierIO_.getClassifierFiles();
    for (int i = 0; i < files.size(); ++i)
    {
        cerr << "files[" << i << "] = " << files[i] << endl;
    }
    if (files.size() != 4)
    {
        cerr << "No files to load classifier" << endl;
        return false;
    }
		detector_ = new GPU_NNDetect(files[0], files[1], files[2], files[3]);

	// Verfiy the load
	if( !detector_ || !detector_->initialized() )
	{
		cerr << "Error loading GPU_NNDetect" << endl;
		return false;
	}
	reload_ = false;
	return true;
}

void DetectState::toggleGPU(void)
{
   gpu_ = !gpu_;
   reload_ = true;
}

void DetectState::changeSubModel(bool increment)
{
   if (classifierIO_.findNextClassifierStage(increment))
	  reload_ = true;
}

void DetectState::changeModel(bool increment)
{
   if (classifierIO_.findNextClassifierDir(increment))
	  reload_ = true;
}

std::string DetectState::print(void) const
{
   return classifierIO_.print();
}
