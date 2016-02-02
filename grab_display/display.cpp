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

using namespace std;
using namespace cv;

#if 1
//Values for purple screen:
int g_h_max = 170;
int g_h_min = 130;
int g_s_max = 255;
int g_s_min = 147;
int g_v_max = 255;
int g_v_min = 48;
#else
//Values for blue screen:
int g_h_max = 120;
int g_h_min = 110;
int g_s_max = 255;
int g_s_min = 220;
int g_v_max = 150;
int g_v_min = 50;
#endif
int    g_files_per  = 9;
int    g_num_frames = 50;
int    g_min_resize = 0;
int    g_max_resize = 40;
float  g_noise      = 5.0;
string g_outputdir  = ".";

const int min_area = 7000;

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

        Mat   temp;
        Mat   tempm;
        Mat   gframe;
        Mat   variancem;
        float variance;
        int   frame_counter;

        typedef pair<float, int>   Blur_Entry;
        vector<Blur_Entry> lblur;

        for (frame_counter = 0; frame_video.read(frame); frame_counter += 1)
        {
            Rect bounding_rect;
            cvtColor(frame, hsv_input, CV_BGR2HSV);
            if (FindRect(hsv_input, bounding_rect))
            {
                cvtColor(frame, gframe, CV_BGR2GRAY);
                Laplacian(gframe, temp, CV_8UC1);
                meanStdDev(temp, tempm, variancem);
                variance = pow(variancem.at<Scalar>(0, 0)[0], 2);
                lblur.push_back(Blur_Entry(variance, frame_counter));
            }
        }
        sort(lblur.begin(), lblur.end(), greater<Blur_Entry>());
        cout << "Read " << lblur.size() << " valid frames from video of " << frame_counter << " total" << endl;

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
            int dilation_type = MORPH_ELLIPSE;
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
            threshold(btrack, btrack, 128, 255, THRESH_BINARY);
            bounding_rect = AdjustRect(bounding_rect, 1.0);
#ifdef DEBUG
            imshow("Btrack", btrack);
#endif
            Mat hsvframe;
            cvtColor(frame, hsvframe, CV_BGR2HSV);
#ifdef DEBUG
            imshow("HSV", hsvframe);
#endif
            hsvframe.convertTo(hsvframe, CV_16UC3);

            /*add(hsvframe, Scalar(hueAdjust, 0, 0), hsvframe, btrack);
             * for (int l = 0; l < hsvframe.rows; l++)
             * {
             *  for (int m = 0; m < hsvframe.cols; m++)
             *  {
             *      hsvframe.at<Vec3b>(l, m)[0] = hsvframe.at<Vec3b>(l, m)[0] % 180;
             *  }
             * }*/
            hsvframe.convertTo(hsvframe, CV_32FC3);
            Mat splitMat[3];
            /*Possible alt method of adjusting hue
            split(hsvframe, splitMat);
            double min, max;
            minMaxLoc(splitMat[0], &min, &max, NULL, NULL, btrack);
            int step = (179 - max + min) / 8.;*/
            for (int hueAdjust = 0; hueAdjust <= 160; hueAdjust += 20)
            {
                add(hsvframe, Scalar(hueAdjust, 0, 0), hsvframe, btrack);
                for (int l = 0; l < hsvframe.rows; l++)
                {
                    for (int m = 0; m < hsvframe.cols; m++)
                    {
                        hsvframe.at<Vec3b>(l, m)[0] = hsvframe.at<Vec3b>(l, m)[0] % 180;
                    }
                }
                Mat noise = Mat(hsvframe.size(), CV_32F);
                randn(noise, 0.0, g_noise);
                split(hsvframe, splitMat);
                //subtract(splitMat[0], Scalar(min - hueAdjust), splitMat[0], btrack);
                for (int i = 1; i <= 2; i++)
                {
                    double min, max;
                    minMaxLoc(splitMat[i], &min, &max, NULL, NULL, btrack);
                    add(splitMat[i], noise, splitMat[i], btrack);
                    normalize(splitMat[i], splitMat[i], min, max, NORM_MINMAX, -1, btrack);
                }
                Mat hsv_final;
                merge(splitMat, 3, hsv_final);
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
                {
                    scale_adder = g_max_resize + 1;
                }
                else
                {
                    scale_adder = (g_max_resize - g_min_resize) / (g_files_per - 1);
                }

                for (double scale_up = g_min_resize; scale_up <= g_max_resize; scale_up += scale_adder)
                {
                    if (RescaleRect(bounding_rect, final_rect, hsv_input, scale_up))
                    {
                        stringstream write_name;
                        write_name << g_outputdir << "/" + Behead(vid_name) << "_" << setw(5) << setfill('0') << this_frame;
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
