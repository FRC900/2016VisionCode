#include <iostream>
#include <opencv2/opencv.hpp>

#include "zedin.hpp"
#include "zmsout.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		cout << argv[1] << " input output" << endl;
		return 0;
	}
	ZedIn  in(argv[1], false);
	ZMSOut out(argv[2]);

	Mat image;
	Mat depth;
	while (in.update() && in.getFrame(image, depth) )
	{
		out.sync();
		out.saveFrame(image, depth);
	}
	out.sync();
	return 0;
}
