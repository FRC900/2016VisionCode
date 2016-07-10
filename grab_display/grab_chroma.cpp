#include <opencv2/opencv.hpp>

#include <iostream>
#include <iomanip>
#include <functional>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <string>

#include "chroma_key.hpp"

using namespace std;
using namespace cv;
//#define DEBUG

#if 1
//Values for purple screen:
static int g_h_max = 170;
static int g_h_min = 130;
static int g_s_max = 255;
static int g_s_min = 147;
static int g_v_max = 255;
static int g_v_min = 48;
#else
//Values for blue screen:
static int g_h_max = 120;
static int g_h_min = 110;
static int g_s_max = 255;
static int g_s_min = 220;
static int g_v_max = 150;
static int g_v_min = 50;
#endif
static int    g_files_per  = 1; // no resizing for now
static int    g_num_frames = 50;
static int    g_min_resize = 0;
static int    g_max_resize = 0; //no resizing for now
static float  g_noise      = 5.0;
static string g_outputdir  = ".";

#ifdef __CYGWIN__
inline int
stoi(const wstring& __str, size_t *__idx = 0, int __base = 10)
{
    return __gnu_cxx::__stoa<long, int>(&std::wcstol, "stoi", __str.c_str(),
                                        __idx, __base);
}
#endif

template<typename T>
string IntToHex(T i)
{
    stringstream stream;

    stream << setfill('0') << setw(2) << hex << i;
    return stream.str();
}


string Behead(string my_string)
{
    size_t found = my_string.rfind("/");

    return my_string.substr(found + 1);
}


// Resizes a rectangle to a new size, keeping
// it centered on the same point
Rect ResizeRect(const Rect& rect, const Size& size)
{
    Point tl = rect.tl();

    tl.x = tl.x - ((double)size.width - rect.width) / 2;
    tl.y = tl.y - ((double)size.height - rect.height) / 2;
    Point br = rect.br();
    br.x = br.x + ((double)size.width - rect.width) / 2;
    br.y = br.y + ((double)size.height - rect.height) / 2;

    return Rect(tl, br);
}


Rect AdjustRect(const Rect& frame, float ratio)
{
    // adjusts the size of the rectangle to a fixed aspect ratio
    int width  = frame.width;
    int height = frame.height;

    if (width / ratio > height)
    {
        height = width / ratio;
    }
    else if (width / ratio < height)
    {
        width = height * ratio;
    }

    return ResizeRect(frame, Size(width, height));
}


bool RescaleRect(const Rect& the_rect, Rect& output_rect, const Mat& image_cool, double scale_up)
{
    // takes the rect the_rect and resizes it larger by 1+scale_up percent
    // outputs resized rect in output_rect
    int width  = the_rect.width * (1.0 + scale_up / 100.0);
    int height = the_rect.height * (1.0 + scale_up / 100.0);

    output_rect = ResizeRect(the_rect, Size(width, height));

    if ((output_rect.x < 0) || (output_rect.y < 0) ||
        (output_rect.br().x > image_cool.cols) || (output_rect.br().y > image_cool.rows))
    {
        cout << "Rectangle out of bounds!" << endl;
        return false;
    }
    return true;
}


void usage(char *argv[])
{
    cout << "usage: " << argv[0] << " [-r RGBM RGBT] [-f frames] [-i files] [--min min] [--max max] filename1 [filename2...]" << endl << endl;
    cout << "-r         RGBM and RGBT are hex colors RRGGBB; M is the median value and T is the threshold above or below the median" << endl;
    cout << "-f         frames is the number of frames grabbed from a video" << endl;
    cout << "-i         files is the number of output image files per frame" << endl;
    cout << "--min      min is the minimum percentage (as a decimal) for resizing for detection" << endl;
    cout << "--max      max is the max percentage (as a decimal) for resizing for detection" << endl;
    cout << "-o         change output directory from cwd/images to [option]/images" << endl;
}


vector<string> Arguments(int argc, char *argv[])
{
    size_t temp_pos;
    int    temp_int;

    vector<string> vid_names;
    vid_names.push_back("");
    if (argc < 2)
    {
        usage(argv);
    }
    else if (argc == 2)
    {
        vid_names[0] = argv[1];
    }
    else
    {
        for (int i = 0; i < argc; i++)
        {
            if (strncmp(argv[i], "-r", 2) == 0)
            {
                try
                {
                    stoi(argv[i + 1], &temp_pos, 16);
                    if (temp_pos != 6)
                    {
                        cout << "Wrong number of hex digits for param -r!" << endl;
                        break;
                    }
                    stoi(argv[i + 2], &temp_pos, 16);
                    if (temp_pos != 6)
                    {
                        cout << "Wrong number of hex digits for param -r!" << endl;
                        break;
                    }
                }
                catch (...)
                {
                    usage(argv);
                    break;
                }
                temp_int  = stoi(argv[i + 1], &temp_pos, 16);
                g_h_min   = temp_int / 65536;
                temp_int -= g_h_min * 65536;
                g_s_min   = temp_int / 256;
                temp_int -= g_s_min * 256;
                g_v_min   = temp_int;
                temp_int  = stoi(argv[i + 2], &temp_pos, 16);
                g_h_max   = temp_int / 65536;
                temp_int -= g_h_min * 65536;
                g_s_max   = temp_int / 256;
                temp_int -= g_s_min * 256;
                g_v_max   = temp_int;
                i        += 2;
            }
            else if (strncmp(argv[i], "-f", 2) == 0)
            {
                try
                {
                    if (stoi(argv[i + 1]) < 1)
                    {
                        cout << "Must get at least one frame per file!" << endl;
                        break;
                    }
                }
                catch (...)
                {
                    usage(argv);
                    break;
                }
                g_num_frames = stoi(argv[i + 1]);
                i           += 1;
            }
            else if (strncmp(argv[i], "--min", 4) == 0)
            {
                try
                {
                    if (stoi(argv[i + 1]) < 0)
                    {
                        cout << "Cannot resize below 0%!" << endl;
                        break;
                    }
                }
                catch (...)
                {
                    usage(argv);
                    break;
                }
                g_min_resize = stoi(argv[i + 1]);
                i++;
            }
            else if (strncmp(argv[i], "--max", 4) == 0)
            {
                try
                {
                    stoi(argv[i + 1]);
                }
                catch (...)
                {
                    usage(argv);
                    break;
                }
                g_max_resize = stoi(argv[i + 1]);
                i++;
            }
            else if (strncmp(argv[i], "-i", 2) == 0)
            {
                try
                {
                    if (stoi(argv[i + 1]) < 1)
                    {
                        cout << "Must output at least 1 file per frame!" << endl;
                        break;
                    }
                }
                catch (...)
                {
                    usage(argv);
                    break;
                }
                g_files_per = stoi(argv[i + 1]);
                i++;
            }
            else if (strncmp(argv[i], "-o", 2) == 0)
            {
                g_outputdir = argv[i + 1];
                i++;
            }
            else if (argv[i] != argv[0])
            {
                for ( ; i < argc; i++)
                {
                    if (vid_names[0] == "")
                    {
                        vid_names[0] = argv[i];
                    }
                    else
                    {
                        vid_names.push_back(argv[i]);
                    }
                }
            }
        }
    }
    return vid_names;
}


typedef pair<float, int> Blur_Entry;
void readVideoFrames(const string &vidName, int &frameCounter, vector<Blur_Entry> &lblur)
{
	VideoCapture frameVideo(vidName);

	lblur.clear();
	frameCounter = 0;
	if (!frameVideo.isOpened())
	{
		return;
	}

	Mat frame;
	Mat hsvInput;

	Mat temp;
	Mat tempm;
	Mat gframe;
	Mat variancem;

#ifndef DEBUG
	// Grab a list of frames which have an identifiable
	// object in them.  For each frame, compute a
	// blur score indicating how clear each frame is
	for (frameCounter = 0; frameVideo.read(frame); frameCounter += 1)
	{
		cvtColor(frame, hsvInput, CV_BGR2HSV);
		Rect bounding_rect;
		if (FindRect(hsvInput, Scalar(g_h_min, g_s_min, g_v_min), Scalar(g_h_max, g_s_max, g_v_max), bounding_rect))
		{
			cvtColor(frame, gframe, CV_BGR2GRAY);
			Laplacian(gframe, temp, CV_8UC1);
			meanStdDev(temp, tempm, variancem);
			float variance = pow(variancem.at<Scalar>(0, 0)[0], 2);
			lblur.push_back(Blur_Entry(variance, frameCounter));
		}
	}
#else
	frameCounter = 1;
	lblur.push_back(Blur_Entry(1,137));
#endif
	sort(lblur.begin(), lblur.end(), greater<Blur_Entry>());
	cout << "Read " << lblur.size() << " valid frames from video of " << frameCounter << " total" << endl;
}


int main(int argc, char *argv[])
{
    vector<string> vid_names = Arguments(argc, argv);
    if (vid_names[0] == "")
    {
        cout << "Invalid program syntax!" << endl;
        return 0;
    }
#ifdef DEBUG
    namedWindow("Original", WINDOW_AUTOSIZE);
    namedWindow("RangeControl", WINDOW_AUTOSIZE);
    namedWindow("Tracking", WINDOW_AUTOSIZE);

    createTrackbar("HueMin", "RangeControl", &g_h_min, 255);
    createTrackbar("HueMax", "RangeControl", &g_h_max, 255);

    createTrackbar("SatMin", "RangeControl", &g_s_min, 255);
    createTrackbar("SatMax", "RangeControl", &g_s_max, 255);

    createTrackbar("ValMin", "RangeControl", &g_v_min, 255);
    createTrackbar("ValMax", "RangeControl", &g_v_max, 255);
#endif

	RNG rng(time(NULL));

	// Middle of chroma-key range
    Vec3b mid((g_h_min + g_h_max) / 2, (g_s_min + g_s_max) / 2, (g_v_min + g_v_max) / 2);

    for (auto vidName = vid_names.cbegin(); vidName != vid_names.cend(); ++vidName)
    {
        cout << *vidName << endl;

        Mat frame;
        Mat hsvframe;
		Mat objMask;
		Rect bounding_rect;

        int   frame_counter;

        vector<Blur_Entry> lblur;
		readVideoFrames(*vidName, frame_counter, lblur);
		if (lblur.empty())
        {
            cout << "Capture not open; invalid video" << endl;
            continue;
        }

        VideoCapture frame_video(*vidName);
        int          frame_count = 0;
        vector<bool> frame_used(frame_counter);
        const int    frame_range = 10;      // Try to space frames out by this many unused frames
        for (auto it = lblur.begin(); (frame_count < g_num_frames) && (it != lblur.end()); ++it)
        {
            // Check to see that we haven't used a frame close to this one
            // already - hopefully this will give some variety in the frames
            // which are used
            int  this_frame      = it->second;
            bool frame_too_close = false;
            for (int j = max(this_frame - frame_range + 1, 0); !frame_too_close && (j < min((int)frame_used.size(), this_frame + frame_range)); j++)
            {
                if (frame_used[j])
                {
                    frame_too_close = true;
                }
            }

            if (frame_too_close)
            {
                continue;
            }

            frame_used[this_frame] = true;

            frame_video.set(CV_CAP_PROP_POS_FRAMES, this_frame);
            frame_video >> frame;
            cvtColor(frame, hsvframe, CV_BGR2HSV);
#ifdef DEBUG
			imshow("Frame at read", frame);
			imshow("HSV Frame at read", hsvframe);
#endif

			// Get a mask image. Pixels for the object in question
			// will be set to 255, others to 0
			if (!getMask(hsvframe, Scalar(g_h_min, g_s_min, g_v_min), Scalar(g_h_max, g_s_max, g_v_max), objMask, bounding_rect))
			{
				continue;
			}
            bounding_rect = AdjustRect(bounding_rect, 1.0);

			// Fill non-mask pixels with midpoint
			// of chroma-key color
            for (int k = 0; k < objMask.rows; k++)
            {
                for (int l = 0; l < objMask.cols; l++)
                {
                    uchar point = objMask.at<uchar>(k, l);
                    if (point == 0)
                    {
                        hsvframe.at<Vec3b>(k, l) = mid;
                    }
                }
            }
#ifdef DEBUG
            imshow("objMask returned from getMask", objMask);
            imshow("HSV frame after fill with mid", hsvframe);
#endif
            /*hsvframe.convertTo(hsvframe, CV_16UC3);
             *add(hsvframe, Scalar(hueAdjust, 0, 0), hsvframe, objMask);
             * for (int l = 0; l < hsvframe.rows; l++)
             * {
             *  for (int m = 0; m < hsvframe.cols; m++)
             *  {
             *      hsvframe.at<Vec<short,3>>(l, m)[0] = hsvframe.at<Vec<short,3>>(l, m)[0] % 180;
             *  }
             * }*/
            hsvframe.convertTo(hsvframe, CV_32FC3);
            Mat splitMat[3];
            /*Possible alt method of adjusting hue
            split(hsvframe, splitMat);
            double min, max;
            minMaxLoc(splitMat[0], &min, &max, NULL, NULL, objMask);
            int step = (179 - max + min) / 8.;*/
            for (int hueAdjust = 0; hueAdjust <= 160; hueAdjust += 30)
            {
				int rndHueAdjust = max(hueAdjust + rng.uniform(-10,10), 0);
                add(hsvframe, Scalar(rndHueAdjust, 0, 0), hsvframe, objMask);
                for (int l = 0; l < hsvframe.rows; l++)
                {
                    for (int m = 0; m < hsvframe.cols; m++)
                    {
						float val = hsvframe.at<Vec3f>(l, m)[0];
						if (val >= 180.)
							hsvframe.at<Vec3f>(l, m)[0] = val - 180.0;
                    }
                }
                Mat noise = Mat::zeros(hsvframe.size(), CV_32F);
                randn(noise, 0.0, g_noise);
                split(hsvframe, splitMat);
                //subtract(splitMat[0], Scalar(min - hueAdjust), splitMat[0], objMask);
                for (int i = 1; i <= 2; i++)
                {
                    double min, max;
                    minMaxLoc(splitMat[i], &min, &max, NULL, NULL, objMask);
                    add(splitMat[i], noise, splitMat[i], objMask);
                    normalize(splitMat[i], splitMat[i], min, max, NORM_MINMAX, -1, objMask);
                }
                Mat hsv_final;
                merge(splitMat, 3, hsv_final);
                hsv_final.convertTo(hsv_final, CV_8UC3);
                cvtColor(hsv_final, frame, CV_HSV2BGR);
#ifdef DEBUG
                imshow("Final HSV", hsv_final);
                imshow("Final RGB", frame);
                waitKey(0);
#endif

				int fail_count = 0;
                for (int i = 0; (i < g_files_per) && (fail_count < 100); )
                {
					double scale_up = rng.uniform((double)g_min_resize, g_max_resize+1.0);
					Rect final_rect;
                    if (RescaleRect(bounding_rect, final_rect, frame, scale_up))
                    {
                        stringstream write_name;
                        write_name << g_outputdir << "/" + Behead(*vidName) << "_" << setw(5) << setfill('0') << this_frame;
                        write_name << "_" << setw(4) << final_rect.x;
                        write_name << "_" << setw(4) << final_rect.y;
                        write_name << "_" << setw(4) << final_rect.width;
                        write_name << "_" << setw(4) << final_rect.height;
                        write_name << "_" << setw(3) << rndHueAdjust;
                        write_name << ".png";
                        imwrite(write_name.str().c_str(), frame(final_rect));
						i++;
						fail_count = 0;
                    }
					else
						fail_count += 1;
                }
            }
            frame_count += 1;
        }

        /*for(int j = 0; j < g_num_frames; j++)
         * {
         *  cout << lblur[j] << ",";
         * }
         * cout << endl;
         * for(int j = 0; j < g_num_frames; j++)
         * {
         *  cout << frame_holder[j] << ",";
         * }
         * cout << endl;*/
    }
    cout << "0x" << IntToHex((g_h_min + g_h_max) / 2) << IntToHex((g_s_min + g_s_max) / 2) << IntToHex((g_v_min + g_v_max) / 2);
    cout << " 0x" << IntToHex((g_h_min + g_h_max) / 2 - g_h_min) << IntToHex((g_s_min + g_s_max) / 2 - g_s_min) << IntToHex((g_v_min + g_v_max) / 2 - g_v_min) << endl;
    return 0;
}
