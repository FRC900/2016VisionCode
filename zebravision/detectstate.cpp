#include <iostream>
#include <string>

#include "detectstate.hpp"
#include "CaffeClassifier.hpp"
#include "GIEClassifier.hpp"

using namespace std;
using namespace cv;

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
	reload_(true)
{
   update();
}

DetectState::~DetectState()
{
	if (detector_)
		delete detector_;
}

bool DetectState::update(void)
{
   if (reload_ == false)
	  return true;

	if (detector_)
	   delete detector_;

    vector<string> d12Files = d12IO_.getClassifierFiles();
    for (size_t i = 0; i < d12Files.size(); ++i)
    {
        cerr << "D12Files[" << i << "] = " << d12Files[i] << endl;
    }
    if (d12Files.size() != 4)
    {
        cerr << "No Files to load classifier" << endl;
        return false;
    }

    vector<string> d24Files = d24IO_.getClassifierFiles();
    for (size_t i = 0; i < d24Files.size(); ++i)
    {
        cerr << "D24Files[" << i << "] = " << d24Files[i] << endl;
    }
    if (d24Files.size() != 4)
    {
        cerr << "No Files to load classifier" << endl;
        return false;
    }

    vector<string> c12Files = c12IO_.getClassifierFiles();
    for (size_t i = 0; i < c12Files.size(); ++i)
    {
        cerr << "C12Files[" << i << "] = " << c12Files[i] << endl;
    }
    if (c12Files.size() != 4)
    {
        cerr << "No Files to load classifier" << endl;
        return false;
    }

    vector<string> c24Files = c24IO_.getClassifierFiles();
    for (size_t i = 0; i < c24Files.size(); ++i)
    {
        cerr << "C24Files[" << i << "] = " << c24Files[i] << endl;
    }
    if (c24Files.size() != 4)
    {
        cerr << "No Files to load classifier" << endl;
        return false;
    }

	// TODO : only reload individual nets if files change?
	if (!gie_)
	{
		if (!gpuClassifier_)
		{
			if (!gpuDetect_)
				detector_ = new ObjDetectCPUCaffeCPU(d12Files, d24Files, c12Files, c24Files, hfov_);
			else
				detector_ = new ObjDetectCPUCaffeGPU(d12Files, d24Files, c12Files, c24Files, hfov_);
		}
		else
		{
			if (!gpuDetect_)
				detector_ = new ObjDetectGPUCaffeCPU(d12Files, d24Files, c12Files, c24Files, hfov_);
			else
				detector_ = new ObjDetectGPUCaffeGPU(d12Files, d24Files, c12Files, c24Files, hfov_);
		}
	}
	else
	{
		// GIE implies GPU detection - CPU doesn't make sense there
		if (!gpuClassifier_)
		{
			detector_ = new ObjDetectCPUGIEGPU(d12Files, d24Files, c12Files, c24Files, hfov_);
		}
		else
		{
			detector_ = new ObjDetectGPUGIEGPU(d12Files, d24Files, c12Files, c24Files, hfov_);
		}
	}

	// Verfiy the load
	if( !detector_ || !detector_->initialized() )
	{
		cerr << "Error loading GPU_NNDetect" << endl;
		return false;
	}
	reload_ = false;
	return true;
}

void DetectState::toggleGPUDetect(void)
{
   gpuDetect_ = !gpuDetect_;
   reload_ = true;
}

void DetectState::toggleGPUClassifier(void)
{
   gpuClassifier_ = !gpuClassifier_;
   reload_ = true;
}
void DetectState::toggleGIE(void)
{
   gie_ = !gie_;
   reload_ = true;
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
	if (gpuClassifier_)
		ret += "G_";
	else
		ret += "C_";
	if (gpuDetect_)
		ret += "G_";
	else
		ret += "C_";
	if (gie_)
		ret += "_GIE";
	else
		ret += "_Caffe";
	ret += " " + d12IO_.print() + "," + d24IO_.print() + "," + c12IO_.print() + "," + c24IO_.print();
	return ret;
}
