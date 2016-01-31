//standard include
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <math.h>
#include <algorithm>
#include <stdint.h>
#include <iostream>

//opencv include
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

class GoalDetector {

public:

GoalDetector();

const float dist_to_goal() { return _dist_to_goal; }
const float angle_to_goal() { return _angle_to_goal; }

bool processFrame(Mat &image, Mat &depth);

private:

vector< Point2f > _goal_shape_contour; //hold the shape of the goal so we can easily get info from it
float _camera_hfov = 84.14 * (M_PI / 180.0); //determined experimentally
float _camera_vfov = 53.836 * (M_PI / 180.0); //determined experimentally
float _goal_height = 2.159;

int _hue_min = 60; //60-95 is a good range for bright green
int _hue_max = 95;
int _sat_min =  180;
int _sat_max = 255;
int _val_min =  67;
int _val_max = 255;

float _dist_to_goal;
float _angle_to_goal;

static void generateThreshold(const Mat &ImageIn, Mat &ImageOut, int H_MIN, int H_MAX, int S_MIN, int S_MAX, int V_MIN, int V_MAX);

static float minOfMat(Mat &img, Mat &mask, bool (*f)(float), bool max=false, int range=10);

static bool countPixel(float v) { if( isnan(v) || v <= 0) { return false; } else { return true; } }

};
