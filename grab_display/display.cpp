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
int g_h_max = 170;
int g_h_min = 130;
int g_s_max = 255;
int g_s_min = 147;
int g_v_max = 255;
int g_v_min = 48;
int g_files_per = 10;
int g_num_frames = 10;
int g_min_resize = 0;
int g_max_resize = 25;
string g_outputdir = ".";
RNG rng(12345);
#ifdef __CYGWIN__
inline string
to_string(int __val)
{ return __gnu_cxx::__to_xstring<string>(vsnprintf, 4 * sizeof(int),
    "%d", __val); }
inline int
stoi(const wstring& __str, size_t* __idx = 0, int __base = 10)
{ return __gnu_cxx::__stoa<long, int>(&std::wcstol, "stoi", __str.c_str(),
                     __idx, __base); }
#endif
template< typename T >
string IntToHex(T i)
{
  stringstream stream;
  stream << setfill ('0') << setw(2) << hex << i;
  return stream.str();
}
string Behead(string my_string)
{
    size_t found = my_string.rfind("/");
    return my_string.substr(found+1);
}
bool FindRect(Mat &frame, Rect &output)
{
        /* prec: frame image &frame, integers 0-255 for each min and max HSV value
        *  postc: a rectangle bounding the image we want
        *  takes the frame image, filters to only find values in the range we want, finds
        *  the counters of the object and bounds it with a rectangle
        */
        Mat btrack;
        inRange(frame, Scalar(g_h_min, g_s_min, g_v_min), Scalar(g_h_max, g_s_max, g_v_max), btrack);
        imshow("BtrackR", btrack);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        vector<Point> points;
        findContours( btrack, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        for( size_t i = 0; i < hierarchy.size(); i++ )
        {
            if(hierarchy[i][3] >= 0 && boundingRect(contours[i]).area() > 1000)
            {
                points = contours[i];
                break;
            }
        }
        if(points.empty())
        {
            return false;
        }
        output = boundingRect(points);
        return true;
}
Rect AdjustRect(Rect &frame, float ratio)
{
    // adjusts the size of the rectangle to a fixed aspect ratio
    Size rect_size = frame.size();
    int width = rect_size.width;
    int height = rect_size.height;
    if (width / ratio > height)
    {
        height = width / ratio;
    }
    else if (width / ratio < height)
    {
        width = height * ratio;
    }
    Point tl = frame.tl();
    tl.x = tl.x - (width - rect_size.width)/2;
    tl.y = tl.y - (height - rect_size.height)/2;
    Point br = frame.br();
    br.x = br.x + (width - rect_size.width)/2;
    br.y = br.y + (height - rect_size.height)/2;
    return Rect(tl, br);
}
bool ResizeRect(Rect &the_rect, Rect &output_rect, const Mat image_cool)
{
    //takes the rect &the_rect and randomly resizes it larger within range (g_min_resize, g_max_resize) outputs the rect
    //to image imageCool
    Point tl = the_rect.tl();
    Point br = the_rect.br();
    if(tl.x < 0 || tl.y < 0 || br.x > image_cool.cols || br.y > image_cool.rows)
    {
        cout << "Rectangle out of bounds!" << endl;
        return false;
    }
    tl = Point(-1,-1);
    while (tl.x < 0 || tl.y < 0 || br.x > image_cool.cols || br.y > image_cool.rows)
    {
        float adjust = rng.uniform(g_min_resize,g_max_resize);
        Size rect_size = the_rect.size();
        float width = rect_size.width * (1 + adjust/100);
        float height = rect_size.height * (1 + adjust/100);
        tl = the_rect.tl();
        tl.x = tl.x - (width - rect_size.width)/2;
        tl.y = tl.y - (height - rect_size.height)/2;
        br = the_rect.br();
        br.x = br.x + (width - rect_size.width)/2;
        br.y = br.y + (height - rect_size.height)/2;
    }
    output_rect = Rect(tl, br);
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
    int temp_int;
    vector<string> vid_names;
    vid_names.push_back("");
    if(argc < 2)
    {
        usage(argv);
    }
    else if(argc == 2)
    {
        vid_names[0] = argv[1];
    }
    else
    {
        for(int i = 0; i < argc; i++)
        {
            if(strncmp(argv[i], "-r", 2) == 0)
            {
                try
                {
                    stoi(argv[i+1], &temp_pos, 16);
                    if(temp_pos != 6)
                    {
                        cout << "Wrong number of hex digits for param -r!" << endl;
                        break;
                    }
                    stoi(argv[i+2], &temp_pos, 16);
                    if(temp_pos != 6)
                    {
                        cout << "Wrong number of hex digits for param -r!" << endl;
                        break;
                    }
                }
                catch(...)
                {
                    usage(argv);
                    break;
                }
                temp_int = stoi(argv[i+1], &temp_pos, 16);
                g_h_min = temp_int/65536;
                temp_int -= g_h_min*65536;
                g_s_min = temp_int/256;
                temp_int -= g_s_min*256;
                g_v_min = temp_int;
                temp_int = stoi(argv[i+2], &temp_pos, 16);
                g_h_max = temp_int/65536;
                temp_int -= g_h_min*65536;
                g_s_max = temp_int/256;
                temp_int -= g_s_min*256;
                g_v_max = temp_int;
                i += 2;
            }
            else if(strncmp(argv[i],"-f",2) == 0)
            {
                try
                {
                    if(stoi(argv[i+1]) < 1)
                    {
                        cout << "Must get at least one frame per file!" << endl;
                        break;
                    }
                }
                catch(...)
                {
                    usage(argv);
                    break;
                }
                g_num_frames = stoi(argv[i+1]);
                i += 1;
            }
            else if(strncmp(argv[i],"--min",4) == 0)
            {
                try
                {
                    if(stoi(argv[i+1]) < 1)
                    {
                        cout << "Cannot resize below 0%!" << endl;
                        break;
                    }
                }
                catch(...)
                {
                    usage(argv);
                    break;
                }
                g_min_resize = stoi(argv[i+1]);
                i++;
            }
            else if(strncmp(argv[i], "--max", 4) == 0)
            {
                try
                {
                    stoi(argv[i+1]);
                }
                catch(...)
                {
                    usage(argv);
                    break;
                }
                g_max_resize = stoi(argv[i+1]);
                i++;
            }
            else if(strncmp(argv[i], "-i", 2) == 0)
            {
                try
                {
                    if(stoi(argv[i+1]) < 1)
                    {
                        cout << "Must output at least 1 file per frame!" << endl;
                        break;
                    }
                }
                catch(...)
                {
                    usage(argv);
                    break;
                }
                g_files_per = stoi(argv[i+1]);
                i++;
            }
            else if(strncmp(argv[i], "-o", 2) == 0)
            {
                g_outputdir = argv[i+1];
                i++;
            }
            else if(argv[i] != argv[0])
            {
                for(; i < argc; i++)
                {
                    if(vid_names[0] == "")
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
int main(int argc, char *argv[]) {
    vector<string> vid_names = Arguments(argc, argv);
    if(vid_names[0] == "")
    {
        cout << "Invalid program syntax!" << endl;
        return 0;
    }
    namedWindow("Original", WINDOW_AUTOSIZE);
    namedWindow("RangeControl", WINDOW_AUTOSIZE);
    namedWindow("Tracking", WINDOW_AUTOSIZE);

    createTrackbar("HueMin","RangeControl", &g_h_min,255);
    createTrackbar("HueMax","RangeControl", &g_h_max,255);

    createTrackbar("SatMin","RangeControl", &g_s_min,255);
    createTrackbar("SatMax","RangeControl", &g_s_max,255);

    createTrackbar("ValMin","RangeControl", &g_v_min,255);
    createTrackbar("ValMax","RangeControl", &g_v_max,255);

    String vid_name = "";
    Mat mid = Mat_<Vec3b>(1,1) << Vec3b((g_h_min + g_h_max)/2, (g_s_min + g_s_max)/2, (g_v_min + g_v_max)/2);
    cvtColor(mid, mid, CV_HSV2BGR);
    for(int i = 0; i < vid_names.size(); i++)
    {
        vid_name = vid_names[i];
        cout << vid_name << endl;
        VideoCapture frame_video(vid_name);

        if(!frame_video.isOpened())
        {
            cout << "Capture not open; invalid video" << endl;
            continue;
        }

        Mat frame;
        Mat hsv_input;

        vector<Mat> channels;
        vector<Mat> temp2(3);

        int count = 0;
        bool isColor = true;

        Mat temp;
        Mat hue;
        Mat sat;
        Mat val;
        Mat rgbVal;
        Mat ret;
        Mat gframe;
        Mat variancem;
        Mat tempm;
        float variance;
        vector<float> lblur(g_num_frames);
        int frame_counter = 0;
        int frame_holder[g_num_frames];
        String write_name = "";
        while(1) {
            frame_counter = frame_video.get(CV_CAP_PROP_POS_MSEC);
            frame_video.read(frame);
            if(frame.empty())
            {
                break;
            }
            frame_counter++;
            bool exists;
            Rect bounding_rect;
            Rect temp_rect;
            cvtColor(frame, hsv_input, CV_BGR2HSV);
            exists = FindRect(hsv_input, bounding_rect);
            if(exists == false)
            {
                continue;
            }
            bounding_rect = AdjustRect(bounding_rect, 1.0);
            exists = ResizeRect(bounding_rect, temp_rect, hsv_input);
            if(exists == false)
            {
                continue;
            }
            cvtColor( frame, gframe, CV_BGR2GRAY);
            Laplacian(gframe, temp, CV_8UC1);
            meanStdDev(temp, tempm, variancem);
            variance = pow(variancem.at<Scalar>(0,0)[0], 2);
            int min_pos = distance(lblur.begin(), min_element(lblur.begin(), lblur.end()));
            if (variance > lblur[min_pos])
            {
                lblur[min_pos] = variance;
                frame_holder[min_pos] = frame_counter;
            }
        }
        for(int j = 0; j < g_num_frames; j++)
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
            vector<Vec4i> hierarchy;
            int contour_index;
            Mat btrack_cp;
            btrack_cp = btrack;
            findContours( btrack_cp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
            for( size_t i = 0; i < hierarchy.size(); i++ )
            {
                if(hierarchy[i][3] >= 0 && boundingRect(contours[i]).area() > 1000)
                {
                    contour_index = i;
                    break;
                }
            }
            drawContours(btrack, contours, contour_index, Scalar(255), CV_FILLED);
            int dilation_type = MORPH_RECT;
            int dilation_size = 1;
            Mat element = getStructuringElement( dilation_type,
                                             Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                             Point( dilation_size, dilation_size ) );
            dilate(btrack, btrack, element);
            int erosion_size = 5;
            element = getStructuringElement( dilation_type,
                                             Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                             Point( erosion_size, erosion_size ) );
            erode(btrack, btrack, element );
            for(int k = 0; k < btrack.rows; k++)
            {
                for(int l = 0; l < btrack.cols; l++)
                {
                    uchar point = btrack.at<uchar>(k,l);
                    if(point == 0)
                    {
                        frame.at<Vec3b>(k,l) = mid.at<Vec3b>(0,0);
                    }
                }
            }
            imshow("Finished", frame);
            bounding_rect = AdjustRect(bounding_rect, 1.0);
            for(int k = 0; k < g_files_per; k++)
            {
                ResizeRect(bounding_rect, final_rect, hsv_input);
                vid_name = Behead(vid_name);
                write_name = g_outputdir + "/images/" + vid_name + "_" + to_string(j) + "_" + to_string(k) + ".png";
                imshow("Edges", frame(final_rect));
                imwrite(write_name, frame(final_rect));
            }
        }
        /*for(int j = 0; j < g_num_frames; j++)
        {
            cout << lblur[j] << ",";
        }
        cout << endl;
        for(int j = 0; j < g_num_frames; j++)
        {
            cout << frame_holder[j] << ",";
        }
        cout << endl;*/
    }
    cout << "0x" << IntToHex((g_h_min + g_h_max)/2) << IntToHex((g_s_min + g_s_max)/2) << IntToHex((g_v_min + g_v_max)/2);
    cout << " 0x" << IntToHex((g_h_min + g_h_max)/2 - g_h_min) << IntToHex((g_s_min + g_s_max)/2 - g_s_min) << IntToHex((g_v_min + g_v_max)/2 - g_v_min) << endl;
    return 0;
}
