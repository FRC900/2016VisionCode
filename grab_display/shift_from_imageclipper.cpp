#include <sys/types.h>
#include <dirent.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "imageShift.hpp"

using namespace std;
using namespace cv;

vector<string> &split(const string &s, char delim, vector<string> &elems) {
	stringstream ss(s);
	string item;
	while (getline(ss, item, delim)) {
		if (!item.empty())
			elems.push_back(item);
	}
	return elems;
}

vector<string> split(const string &s, char delim) {
	vector<string> elems;
	split(s, delim, elems);
	return elems;
}
const double targetAR = 1.0;

string srcPath = "/home/kjaget/ball_videos/white_floor/";
string outPath = "shifts";
int main(void)
{
	DIR *dirp = opendir(".");
	struct dirent *dp;
	vector<string> image_names;
	if (!dirp)
		return -1;

	while ((dp = readdir(dirp)) != NULL) 
	{
		if (strstr(dp->d_name, ".png") )
			image_names.push_back(dp->d_name);
	}
	closedir(dirp);
	cout << "Read " << image_names.size() << " image names" << endl;

	createShiftDirs(outPath);
	RNG rng(time(NULL));
	Mat mat;
	for (vector<string>::iterator it = image_names.begin(); it != image_names.end(); ++it)
	{
		int frame;
		int rotation = 0;
		Rect rect;
		*it = it->substr(0, it->rfind('.'));
		vector<string> tokens = split(*it, '_');
		frame       = atoi(tokens[tokens.size()-6].c_str());
		rect.x      = atoi(tokens[tokens.size()-5].c_str());
		rect.y      = atoi(tokens[tokens.size()-4].c_str());
		rect.width  = atoi(tokens[tokens.size()-3].c_str());
		rect.height = atoi(tokens[tokens.size()-2].c_str());

		string inFileName;
		for (size_t i = 0; i < tokens.size() - 6; i++)
		{
			inFileName += tokens[i];
			if (i < tokens.size() - 7)
				inFileName += "_";
		}

		VideoCapture cap((srcPath+inFileName).c_str());
		if( !cap.isOpened() )
		{
			cerr << "Can not open " << srcPath+inFileName << endl;
			continue;
		}
		cap.set(CV_CAP_PROP_POS_FRAMES, frame - 1 );
		cap >> mat;

		const double ar = rect.width / (double)rect.height;
		int added_height = 0;
		int added_width  = 0;
		if (ar > targetAR)
		{
			added_height = rect.width / targetAR - rect.height;
			rect.x -= (added_height/ 2.0) * sin(rotation / 180.0 * M_PI);
			rect.y -= (added_height/ 2.0) * cos(rotation / 180.0 * M_PI);
		}
		else if (ar < targetAR)
		{
			added_width = rect.height * targetAR - rect.width;
			rect.x -= (added_width / 2.0) * cos(rotation / 180.0 * M_PI);
			rect.y += (added_width / 2.0) * sin(rotation / 180.0 * M_PI);
		}
		rect.width  += added_width;
		rect.height += added_height;

		if ((rect.x < 0) || (rect.y < 0) || 
			(rect.br().x >= mat.cols) || (rect.br().y >= mat.rows))
			continue;

		stringstream write_name;
		write_name << inFileName;
		write_name << "_" << setw(5) << setfill('0') << frame;
		write_name << "_" << setw(4) << rect.x;
		write_name << "_" << setw(4) << rect.y;
		write_name << "_" << setw(4) << rect.width;
		write_name << "_" << setw(4) << rect.height;
		write_name << ".png";

		doShifts(mat, rect, rng, Point3f(0,0,1), 10, outPath, write_name.str());
	}
	return 0;
}

