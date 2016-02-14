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

ClassifierIO::ClassifierIO(string baseDir, int dirNum, int stageNum) :
    baseDir_  ( baseDir),
    dirNum_   ( dirNum),
    stageNum_ ( stageNum)
{
    if (!findNextClassifierDir(true))
        cerr << "ERROR: Failed to find first classifier stage" << endl;
}

// using the current directory number, generate a filename for that dir
// if it exists - if it doesnt, return an empty string
string ClassifierIO::getClassifierDir() const
{
    path p(baseDir_ + to_string(dirNum_));
    if (exists(p) && is_directory(p))
    {
        return p.string();
    }
    else
    {
        cerr << "ERROR: Invalid classifier directory: "
             << baseDir_ + to_string(dirNum_) << endl;
        return string();
    }
}

vector<string> ClassifierIO::getClassifierFiles() const
{
    // Get 4 needed files in the following order:
    // 1. deploy.prototxt
    // 2. snapshot_iter_#####.caffemodel
    // 3. mean.binaryproto
    // 4. labels.txt
    vector<string> output;
    /*
    output.push_back("d12/deploy.prototxt");
    output.push_back("d12/network.caffemodel");
    output.push_back("d12/mean.binaryproto");
    output.push_back("d12/labels.txt");
    */

    path classifierPath(getClassifierDir());
    //cerr << "classifier dir=" << classifierPath.string() << endl;

    {
        path tmpPath(classifierPath);
        tmpPath /= "deploy.prototxt";
        if (!exists(tmpPath) || !is_regular_file(tmpPath))
        {
            cerr << "ERROR: Failed to open " << tmpPath.string();
            return output;
        }
        output.push_back(tmpPath.string());
    }

    {
        path tmpPath(classifierPath);
        tmpPath /= "snapshot_iter_" + to_string(stageNum_) + ".caffemodel";
        if (!exists(tmpPath) || !is_regular_file(tmpPath))
        {
            cerr << "ERROR: Failed to open " << tmpPath.string();
            return output;
        }
        output.push_back(tmpPath.string());
    }

    {
        path tmpPath(classifierPath);
        tmpPath /= "mean.binaryproto";
        if (!exists(tmpPath) || !is_regular_file(tmpPath))
        {
            cerr << "ERROR: Failed to open " << tmpPath.string();
            return output;
        }
        output.push_back(tmpPath.string());
    }

    {
        path tmpPath(classifierPath);
        tmpPath /= "labels.txt";
        if (!exists(tmpPath) || !is_regular_file(tmpPath))
        {
            cerr << "ERROR: Failed to open " << tmpPath.string();
            return output;
        }
        output.push_back(tmpPath.string());
    }

    return output;

}

/*
// using the current directory number and stage within that directory,
// generate a filename to load the cascade from.  Check that
// the file exists - if it doesnt, return an empty string
string ClassifierIO::getClassifierName() const
{
   struct stat fileStat;
   stringstream ss;
   string dirName = getClassifierDir();
   if (!dirName.length())
      return string();

   // There are two different incompatible file formats
   // OpenCV uses to store classifier information. For more
   // entertainment value, some are valid for some types of
   // classifiers and not others. Also others break on the GPU
   // version of the code but not the CPU.
   // The net is we need to look for both since depending on
   // the settings we might need one or the other.
   // Here, try the old format first
   ss << dirName << "/cascade_oldformat_" << stageNum_ << ".xml";

   if ((stat(ss.str().c_str(), &fileStat) == 0) && (fileStat.st_size > 5000))
      return string(ss.str());

   // Try the non-oldformat one next
   ss.str(string());
   ss.clear();
   ss << dirName << "/cascade_" << stageNum_ << ".xml";

   if ((stat(ss.str().c_str(), &fileStat) == 0) && (fileStat.st_size > 5000))
      return string(ss.str());

   // Found neither?  Return an empty string
   return string();
}
*/

// Find the next valid classifier. Since some .xml input
// files crash the GPU we've deleted them. Skip over missing
// files in the sequence
bool ClassifierIO::findNextClassifierStage(bool increment)
{
    int adder = increment ? 1 : -1;
    int num = stageNum_ + adder;

    //struct stat fileStat;
    //string dirPath = getClassifierDir();
    path dirPath(getClassifierDir());

   //bool found = false;
   while (num >= 0 && num <= 200000)
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

   while (dnum >= 0 && dnum <= 100 && !found)
   {
       dnum += adder;
       path p(baseDir_ + to_string(dnum));
       if (exists(p) && is_directory(p))
       {
           dirNum_ = dnum;
           found = true;
       }
   }

	 // Try to find a valid classifier in this dir, counting
	 // up from zero
   stageNum_ = 0;
	 if (found && findNextClassifierStage(true))
	 {
	    found = true;
	 }

   return found;
}

string ClassifierIO::print() const
{
   stringstream s;
   s << dirNum_ << ',' << stageNum_;
   return s.str();
}
