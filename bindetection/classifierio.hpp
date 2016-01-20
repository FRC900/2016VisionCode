#ifndef CLASSIFIERIO_HPP__
#define CLASSIFIERIO_HPP__

#include <string>
#include <vector>

using namespace std;

class ClassifierIO
{
   public:
		ClassifierIO(string baseDir, int dirNum, int stageNum);
		string getClassifierDir(void) const;
		string getClassifierName(void) const;
		bool findNextClassifierStage(bool increment);
		bool findNextClassifierDir(bool increment);
                vector<string> getClassifierFiles(void) const;
		string print(void) const;
   private :
		string baseDir_;
		int dirNum_;
		int stageNum_;
};

#endif
