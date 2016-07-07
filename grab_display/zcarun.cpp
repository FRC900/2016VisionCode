#include <string>
#include "zca.hpp"

#include "utilities_common.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	if (argc <= 1)
	{
		cout << "Usage : " << argv[0] << "xml_saved_file" << endl;
		return 1;
	}
	ZCA zca(argv[1]);

	vector<string> ballImageNames;
#if 0
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_0.avi_00697_0397_0169_0245_0245_159.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_2.avi_00467_0006_0039_0409_0409_111.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_4.avi_00284_0312_0220_0264_0264_094.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_3.avi_00251_0303_0015_0302_0302_051.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_0.avi_00588_0492_0144_0295_0295_000.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_4.avi_00338_0332_0238_0240_0240_143.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_2.avi_00428_0111_0164_0293_0293_086.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_4.avi_00274_0283_0201_0294_0294_000.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_1.avi_00601_0297_0248_0295_0295_086.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_0.avi_01075_0420_0068_0372_0372_152.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_0.avi_00364_0198_0142_0237_0237_050.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_0.avi_01131_0358_0256_0330_0330_007.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_1.avi_00784_0316_0179_0281_0281_000.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_1.avi_00021_0535_0315_0252_0252_004.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_5.avi_00062_0254_0290_0255_0255_002.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_5.avi_00732_0008_0118_0389_0389_129.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_4.avi_00193_0291_0020_0351_0351_093.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_5.avi_00799_0289_0020_0385_0385_003.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_0.avi_00563_0531_0162_0253_0253_117.png");
	ballImageNames.push_back("/home/ubuntu/2016VisionCode/grab_display/images_blue/20160120_4.avi_00168_0325_0003_0308_0308_093.png");

	ballImageNames.push_back("shift/1/20160115_01.avi_00120_0328_0126_0179_0179_000.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00120_0328_0126_0179_0179_000.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00312_0006_0166_0215_0215_024.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00312_0006_0166_0215_0215_024.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00327_0164_0150_0195_0195_144.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00327_0164_0150_0195_0195_144.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00350_0271_0225_0179_0179_143.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00350_0271_0225_0179_0179_143.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00530_0219_0406_0161_0161_039.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00530_0219_0406_0161_0161_039.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00566_0278_0329_0165_0165_115.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00566_0278_0329_0165_0165_115.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00587_0148_0267_0175_0175_000.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00587_0148_0267_0175_0175_000.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00632_0173_0255_0170_0170_022.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00632_0173_0255_0170_0170_022.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00644_0169_0149_0171_0171_116.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00644_0169_0149_0171_0171_116.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00662_0244_0076_0184_0184_053.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00662_0244_0076_0184_0184_053.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00676_0293_0216_0165_0165_009.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00676_0293_0216_0165_0165_009.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00694_0265_0221_0159_0159_000.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00694_0265_0221_0159_0159_000.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00704_0207_0233_0154_0154_027.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00704_0207_0233_0154_0154_027.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00817_0116_0359_0172_0172_098.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00817_0116_0359_0172_0172_098.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00827_0121_0359_0167_0167_055.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00827_0121_0359_0167_0167_055.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00862_0136_0335_0177_0177_060.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00862_0136_0335_0177_0177_060.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00875_0217_0275_0162_0162_119.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00875_0217_0275_0162_0162_119.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00893_0171_0304_0177_0177_123.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00893_0171_0304_0177_0177_123.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00908_0149_0221_0178_0178_065.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00908_0149_0221_0178_0178_065.png_chroma_00000.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00928_0253_0249_0172_0172_110.png0.png");
	ballImageNames.push_back("shift/1/20160115_01.avi_00928_0253_0249_0172_0172_110.png_chroma_00000.png");
	for (auto it = ballImageNames.cbegin(); it != ballImageNames.cend(); ++it)
#endif
    vector<string> filePaths;
    GetFilePaths("/media/kjaget/84CA3305CA32F2D2/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/framegrabber", ".png", filePaths);
    GetFilePaths("/media/kjaget/84CA3305CA32F2D2/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/Framegrabber2", ".png", filePaths, true);
    GetFilePaths("/media/kjaget/84CA3305CA32F2D2/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/generic", ".png", filePaths, true);
    cout << filePaths.size() << " images!" << endl;
	const int seed = 54321;
	RNG rng(seed);
	vector<Mat> images;
	const int nImgs = 15000;
	Mat img; // full image data
	Mat patch; // randomly selected image patch from full image

    for (int nDone = 0; nDone < nImgs; ) 
	{
		// Grab a random image from the list
		size_t idx = rng.uniform(0, filePaths.size());
		img = imread(filePaths[idx]);

		// Pick a random row and column from the image
		int r = rng.uniform(0, img.rows);
		int c = rng.uniform(0, img.cols);

		// Pick a random size as well. Make sure it
		// doesn't extend past the edge of the input
		int s = rng.uniform(0, 2*MIN(MIN(r, img.rows-r), MIN(c, img.cols-c)) + 1 );

		if (s < 28)
			continue;

		Rect rect = Rect(c-s/2, r-s/2, s, s);
		cout << "Using " << filePaths[idx] << rect << endl;

		// Display 3 items :
		//  - original input sub-image
		//  - resized version of that image
		//  - resized and ZCA whitenedversion
		Mat fullRes;
		resize(img(rect), fullRes, Size(256,256));

		// Resize down and then back up to 
		// match what the transform is doing
		// (minus the actual transform)
		Mat resizedImg;
		resize(img(rect), resizedImg, Size(24,24));
		resize(resizedImg, resizedImg, Size(256,256));

		Mat whitenedImg = zca.Transform(img(rect));
		resize(whitenedImg, whitenedImg, Size(256,256));

		imshow("Before", fullRes);
		imshow("Resized", resizedImg);
		imshow("Whitened", whitenedImg);
		waitKey(0);
	}
}
