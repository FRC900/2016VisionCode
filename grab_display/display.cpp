#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <string>

using namespace std;
using namespace cv;

#if 0
//Values for purple screen:
int g_h_max = 170;
int g_h_min = 130;
int g_s_max = 255;
int g_s_min = 147;
int g_v_max = 255;
int g_v_min = 48;  
#else
//Values for blue screen:
int    g_h_max      = 120;
int    g_h_min      = 110;
int    g_s_max      = 255;
int    g_s_min      = 220;
int    g_v_max      = 150;
int    g_v_min      = 50;
#endif
int    g_files_per  = 10;
int    g_num_frames = 10;
int    g_min_resize = 0;
int    g_max_resize = 25;
string g_outputdir  = ".";

const int min_area = 2000;

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


bool FindRect(const Mat& frame, Rect& output)
{
    /* prec: frame image &frame, integers 0-255 for each min and max HSV value
     *  postc: a rectangle bounding the image we want
     *  takes the frame image, filters to only find values in the range we want, finds
     *  the counters of the object and bounds it with a rectangle
     */
    Mat btrack;

    inRange(frame, Scalar(g_h_min, g_s_min, g_v_min), Scalar(g_h_max, g_s_max, g_v_max), btrack);
#ifdef DEBUG
    imshow("BtrackR", btrack);
#endif
    vector<vector<Point> > contours;
    vector<Vec4i>          hierarchy;
    vector<Point>          points;
    findContours(btrack, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    for (size_t i = 0; i < hierarchy.size(); i++)
    {
        if ((hierarchy[i][3] >= 0) && (boundingRect(contours[i]).area() > min_area))
        {
            points = contours[i];
            break;
        }
    }
    if (points.empty())
    {
        return false;
    }
    output = boundingRect(points);
    return true;
}

// Resizes a rectangle to a new size, keeping
// it centered on the same point
Rect ResizeRect(const Rect &rect, const Size &size)
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


bool RescaleRect(const Rect& the_rect, Rect& output_rect, const Mat &image_cool, double scale_up)
{
    // takes the rect the_rect and resizes it larger by 1+scale_up percent
	// outputs resized rect in output_rect
	int width  = the_rect.width  * (1.0 + scale_up / 100.0);
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
                    if (stoi(argv[i + 1]) < 1)
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

    String vid_name = "";
    Mat    mid      = Mat_<Vec3b>(1, 1) << Vec3b((g_h_min + g_h_max) / 2, (g_s_min + g_s_max) / 2, (g_v_min + g_v_max) / 2);
    cvtColor(mid, mid, CV_HSV2BGR);
    for (int i = 0; i < vid_names.size(); i++)
    {
        vid_name = vid_names[i];
        cout << vid_name << endl;
        VideoCapture frame_video(vid_name);

        if (!frame_video.isOpened())
        {
            cout << "Capture not open; invalid video" << endl;
            continue;
        }

        Mat frame;
        Mat hsv_input;

        int  count   = 0;
        bool isColor = true;

        Mat           temp;
        Mat           hue;
        Mat           sat;
        Mat           val;
        Mat           rgbVal;
        Mat           ret;
        Mat           gframe;
        Mat           variancem;
        Mat           tempm;
        float         variance;
        vector<float> lblur(g_num_frames);
        int           frame_counter = 0;
        int           frame_holder[g_num_frames];
        while (1)
        {
            frame_counter = frame_video.get(CV_CAP_PROP_POS_MSEC);
            frame_video.read(frame);
            if (frame.empty())
            {
                break;
            }
            frame_counter++;
            bool exists;
            Rect bounding_rect;
            Rect temp_rect;
            cvtColor(frame, hsv_input, CV_BGR2HSV);
            exists = FindRect(hsv_input, bounding_rect);
            if (exists == false)
            {
                continue;
            }
#if 0
            bounding_rect = AdjustRect(bounding_rect, 1.0);
            exists        = RescaleRect(bounding_rect, temp_rect, hsv_input);
            if (exists == false)
            {
                continue;
            }
#endif
            cvtColor(frame, gframe, CV_BGR2GRAY);
            Laplacian(gframe, temp, CV_8UC1);
            meanStdDev(temp, tempm, variancem);
            variance = pow(variancem.at<Scalar>(0, 0)[0], 2);
            int min_pos = distance(lblur.begin(), min_element(lblur.begin(), lblur.end()));
            if (variance > lblur[min_pos])
            {
                lblur[min_pos]        = variance;
                frame_holder[min_pos] = frame_counter;
            }
        }
        for (int j = 0; j < g_num_frames; j++)
        {
            frame_video.set(CV_CAP_PROP_POS_MSEC, frame_holder[j]);
            frame_video >> frame;
            Rect bounding_rect;
            Rect final_rect;
            cvtColor(frame, hsv_input, CV_BGR2HSV);
            FindRect(hsv_input, bounding_rect);
            Mat btrack;
            inRange(hsv_input, Scalar(g_h_min, g_s_min, g_v_min), Scalar(g_h_max, g_s_max, g_v_max), btrack);
            vector<vector<Point> > contours;
            vector<Vec4i>          hierarchy;
            int contour_index;
            Mat btrack_cp;
            btrack_cp = btrack;
            findContours(btrack_cp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
            for (size_t i = 0; i < hierarchy.size(); i++)
            {
                if ((hierarchy[i][3] >= 0) && (boundingRect(contours[i]).area() > min_area))
                {
                    contour_index = i;
                    break;
                }
            }
            if ((contours.size() == 0) || (contour_index >= contours.size()))
            {
                continue;
            }
            drawContours(btrack, contours, contour_index, Scalar(255), CV_FILLED);
            int dilation_type = MORPH_RECT;
            int dilation_size = 1;
            Mat element       = getStructuringElement(dilation_type,
                                                      Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                                      Point(dilation_size, dilation_size));
            dilate(btrack, btrack, element);
            int erosion_size = 5;
            element = getStructuringElement(dilation_type,
                                            Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                            Point(erosion_size, erosion_size));
            erode(btrack, btrack, element);
            for (int k = 0; k < btrack.rows; k++)
            {
                for (int l = 0; l < btrack.cols; l++)
                {
                    uchar point = btrack.at<uchar>(k, l);
                    if (point == 0)
                    {
                        frame.at<Vec3b>(k, l) = mid.at<Vec3b>(0, 0);
                    }
                }
            }
            for (int hueAdjust = 0; hueAdjust <= 160; hueAdjust += 20)
            {
                threshold(btrack, btrack, 128, 255, THRESH_BINARY);
                bounding_rect = AdjustRect(bounding_rect, 1.0);
#ifdef DEUBG
                imshow("Btrack", btrack);
#endif
                Mat hsv_final;
                cvtColor(frame, hsv_final, CV_BGR2HSV);
#ifdef DEBUG
                imshow("HSV", hsv_final);
#endif
                hsv_final.convertTo(hsv_final, CV_16UC3);
                add(hsv_final, Scalar(hueAdjust, 0, 0), hsv_final, btrack);
                for (int l = 0; l < hsv_final.rows; l++)
                {
                    for (int m = 0; m < hsv_final.cols; m++)
                    {
                        hsv_final.at<Vec3b>(l, m)[0] = hsv_final.at<Vec3b>(l, m)[0] % 180;
                    }
                }
                hsv_final.convertTo(hsv_final, CV_8UC3);
#ifdef DEBUG
                imshow("HSV mod", hsv_final);
#endif
                cvtColor(hsv_final, frame, CV_HSV2BGR);
#ifdef DEBUG
                imshow("Modified", frame);
                waitKey(3000);
#endif
				double scale_adder;
				if (g_files_per == 1)
					scale_adder =  g_max_resize + 1;
				else
					scale_adder = (g_max_resize - g_min_resize) / (g_files_per - 1);

                for (double scale_up = g_min_resize; scale_up <= g_max_resize; scale_up += scale_adder)
                {
                    if (RescaleRect(bounding_rect, final_rect, hsv_input, scale_up))
					{
						stringstream write_name;
						int          frame_num = frame_video.get(CV_CAP_PROP_POS_FRAMES) - 1;
						write_name << g_outputdir << "/" + Behead(vid_name) << "_" << setw(5) << setfill('0') << frame_num;
						write_name << "_" << setw(4) << final_rect.x;
						write_name << "_" << setw(4) << final_rect.y;
						write_name << "_" << setw(4) << final_rect.width;
						write_name << "_" << setw(4) << final_rect.height;
						write_name << "_" << setw(3) << hueAdjust;
						write_name << ".png";
						imwrite(write_name.str().c_str(), frame(final_rect));
					}
                }
            }
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
