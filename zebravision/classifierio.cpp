#include <iostream>
#include <sstream>
#include <string>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <boost/filesystem.hpp>

#include "classifierio.hpp"

using namespace std;
using namespace boost::filesystem;

// Default constructor : takes a baseDir, dirNum and stageNum
// combines baseDir and dirNum to create a path to load config info
// from. stageNum is the training stage in that file.
ClassifierIO::ClassifierIO(string baseDir, int dirNum, int stageNum) :
    baseDir_ (baseDir),
    dirNum_  (dirNum),
    stageNum_(stageNum)
{
	// First try to load the full dir + stage info as is. 
	string outputString;
	if (!createFullPath("snapshot_iter_" + to_string(stageNum_) + ".caffemodel", outputString))
	{
		// See if at least the dir is valid. If so, grab the first stage in there
		if (createFullPath("labels.txt", outputString) && findNextClassifierStage(true))
		{
		}
		// If not, search for any valid directory starting with baseDir
		else if (!findNextClassifierDir(true))
			cerr << "ERROR: Failed to find first classifier stage" << endl;
	}
}

// using the current directory number, generate a filename for that dir
// if it exists - if it doesnt, return an empty string
string ClassifierIO::getClassifierDir() const
{
	string fullDir = baseDir_;
	// Special-case -1 to mean no suffix after directory name.
	if (dirNum_ != -1)
	{
		fullDir += "_" + to_string(dirNum_);
	}
    path p(fullDir);
    if (exists(p) && is_directory(p))
    {
        return p.string();
    }
	cerr << "ERROR: Invalid classifier directory: " << fullDir << endl;
	return string();
}

bool ClassifierIO::createFullPath(const string &fileName, string &output) const
{
	path tmpPath(getClassifierDir());
	tmpPath /= fileName;
	if (!exists(tmpPath) || !is_regular_file(tmpPath))
	{
		cerr << "ERROR: Failed to open " << tmpPath.string() << endl;
		return false;
	}
	output = tmpPath.string();
	return true;
}

vector<string> ClassifierIO::getClassifierFiles() const
{
    // Get 4 needed files in the following order:
    // 1. deploy.prototxt
    // 2. snapshot_iter_#####.caffemodel
    // 3. mean.binaryproto
    // 4. labels.txt
    vector<string> output;
	string outputString;

	if (createFullPath("deploy.prototxt", outputString))
	{
		output.push_back(outputString);

		if (createFullPath("snapshot_iter_" + to_string(stageNum_) + ".caffemodel", outputString))
		{
			output.push_back(outputString);

			if (createFullPath("mean.binaryproto", outputString))
			{
				output.push_back(outputString);

				if (createFullPath("labels.txt", outputString))
				{
					output.push_back(outputString);
				}
			}
		}
	}

    return output;
}

// Find the next valid classifier. Since some .xml input
// files crash the GPU we've deleted them. Skip over missing
// files in the sequence
bool ClassifierIO::findNextClassifierStage(bool increment)
{
    int adder = increment ? 1 : -1;
    int num = stageNum_ + adder;

    path dirPath(getClassifierDir());

   while (num >= 0 && num <= 1000000)
   {
       path p(dirPath);
       p /= "snapshot_iter_" + to_string(num) + ".caffemodel";
       if (exists(p) && is_regular_file(p))
       {
           stageNum_ = num;
           return true;
       }
       num += adder;
   }

   return false;
}

// Find the next valid classifier dir. Start with current stage in that
// directory and work down until a classifier is found
bool ClassifierIO::findNextClassifierDir(bool increment)
{
   int adder = increment ? 1 : -1;
   int dnum = dirNum_;
   bool found = false;

   while (dnum >= -1 && dnum <= 100 && !found)
   {
       dnum += adder;
	   string fullDir = baseDir_;
	   // Special-case -1 to mean no suffix after directory name.
	   if (dnum != -1)
	   {
		   fullDir += "_" + to_string(dnum);
	   }
	   path p(fullDir);
       if (exists(p) && is_directory(p))
       {
           found = true;
       }
   }

   if (found)
   {
	   found = false;
	   int savedStage = stageNum_;
	   // Try to find a valid classifier in this dir, starting
	   // with the current stage from the old dir.  Check current
	   // stage first, then count up, finally count down. If none
	   // are found, restore stage to old setting
	   stageNum_ -= 1;
	   if (findNextClassifierStage(true))
	   {
		   found = true;
	   }
	   else
	   {
		   stageNum_ = savedStage;
		   if (findNextClassifierStage(false))
		   {
			   found = true;
		   }
	   }
	   // If no valid stages found, restore stageNum_
	   // since we're not going to change anything
	   if (!found)
		   stageNum_ = savedStage;
	   // Otherwise set the current dirNum_ to the 
	   // one just found
	   else
           dirNum_ = dnum;
   }

   return found;
}

string ClassifierIO::print() const
{
   return path(getClassifierDir()).filename().string() +  "," + to_string(stageNum_);
}
