//standard include
#include <math.h>

//opencv include
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"


class GoalDetector {

public:

GoalDetector();

float dist_to_goal()const { return _dist_to_goal; } //floor distance to goal in m
float angle_to_goal()const { return _angle_to_goal; } //angle robot has to turn to face goal in degrees

bool processFrame(cv::Mat &image, cv::Mat &depth); //this updates dist_to_goal and angle_to_goal

private:

std::vector< cv::Point2f > _goal_shape_contour; //hold the shape of the goal so we can easily get info from it
float _camera_hfov = 84.14 * (M_PI / 180.0); //determined experimentally
float _camera_vfov = 53.836 * (M_PI / 180.0); //determined experimentally
float _goal_height = 2.159; //in m

int _hue_min = 60; //60-95 is a good range for bright green
int _hue_max = 95;
int _sat_min =  180;
int _sat_max = 255;
int _val_min =  67;
int _val_max = 255;

float _dist_to_goal;
float _angle_to_goal;

void generateThreshold(const cv::Mat &ImageIn, cv::Mat &ImageOut, int H_MIN, int H_MAX, int S_MIN, int S_MAX, int V_MIN, int V_MAX);

float minOfMat(cv::Mat &img, cv::Mat &mask, bool (*f)(float), bool max=false, int range=10);

static bool countPixel(float v) { if( isnan(v) || v <= 0) { return false; } else { return true; } } //small inline function to pass to minOfMat

};
