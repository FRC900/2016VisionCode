#include <iostream>
#include <opencv2/opencv.hpp>

#include "zmsin.hpp"
#include "zmsout.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		cout << argv[1] << " output input1 input2 ... inputN" << endl;
		return 0;
	}
	ZMSOut out(argv[1]);
	Mat image;
	Mat depth;
	for (int i = 2; i < argc; i++)
	{
		ZMSIn in(argv[i]);

		while (in.update() && in.getFrame(image, depth))
		{
			out.sync();
			out.saveFrame(image, depth);
			cout << in.FPS() << " FPS" << endl;
		}
	}
	out.sync();
	return 0;
}
