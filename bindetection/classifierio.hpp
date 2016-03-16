#ifndef CLASSIFIERIO_HPP__
#define CLASSIFIERIO_HPP__

#include <string>
#include <vector>

class ClassifierIO
{
	public:
		ClassifierIO(std::string baseDir, int dirNum, int stageNum);
		std::string getClassifierDir(void) const;
		bool findNextClassifierStage(bool increment);
		bool findNextClassifierDir(bool increment);
		bool createFullPath(const std::string &fileName, std::string &output) const;
		std::vector<std::string> getClassifierFiles(void) const;
		std::string print(void) const;
	private:
		std::string baseDir_;
		int dirNum_;
		int stageNum_;
};

#endif
