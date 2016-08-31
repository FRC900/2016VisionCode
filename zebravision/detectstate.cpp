#include <iostream>
#include <string>

#include "detectstate.hpp"
#include "CaffeClassifier.hpp"
#include "GIEClassifier.hpp"

using namespace std;
using namespace cv;

DetectState::DetectState(const ClassifierIO &d12IO, const ClassifierIO &d24IO, const ClassifierIO &c12IO, const ClassifierIO &c24IO, float hfov, bool gie) :
    detector_(NULL),
	d12IO_(d12IO),
	d24IO_(d24IO),
	c12IO_(c12IO),
	c24IO_(c24IO),
	d12_(NULL),
	d24_(NULL),
	c12_(NULL),
	c24_(NULL),
	hfov_(hfov),
	gie_(gie),
	reload_(true)
{
   update();
}

bool DetectState::update(void)
{
   if (reload_ == false)
	  return true;

	if (detector_)
	   delete detector_;

	if (d12_)
		delete (d12_);
	if (d24_)
		delete (d24_);
	if (c12_)
		delete (c12_);
	if (c24_)
		delete (c24_);

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
		d12_ = new CaffeClassifier<Mat>(d12Files[0], d12Files[1], d12Files[2], d12Files[3], 256);
		d24_ = new CaffeClassifier<Mat>(d24Files[0], d24Files[1], d24Files[2], d24Files[3], 64);
		c12_ = new CaffeClassifier<Mat>(c12Files[0], c12Files[1], c12Files[2], c12Files[3], 64);
		c24_ = new CaffeClassifier<Mat>(c24Files[0], c24Files[1], c24Files[2], c24Files[3], 64);
	}
	else
	{
		d12_ = new GIEClassifier(d12Files[0], d12Files[1], d12Files[2], d12Files[3], 256);
		d24_ = new GIEClassifier(d24Files[0], d24Files[1], d24Files[2], d24Files[3], 64);
		c12_ = new GIEClassifier(c12Files[0], c12Files[1], c12Files[2], c12Files[3], 64);
		c24_ = new GIEClassifier(c24Files[0], c24Files[1], c24Files[2], c24Files[3], 64);
	}
	detector_ = new GPU_NNDetect(d12_, d24_, c12_, c24_, hfov_);

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
   return d12IO_.print() + "," + d24IO_.print() + "," + c12IO_.print() + "," + c24IO_.print();
}
