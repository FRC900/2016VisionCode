//standard include
#include <math.h>
#include <iostream>

//opencv include
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include "Utilities.hpp"
#include "track3d.hpp"

class GoalDetector {
public:

    GoalDetector(cv::Point2f fov_size, cv::Size frame_size);

    float dist_to_goal(void) const { return _goal_found ? _dist_to_goal : -1.0; }   //floor distance to goal in m
    float angle_to_goal(void) const { return _goal_found ? _angle_to_goal : -1.0; } //angle robot has to turn to face goal in degrees

    void processFrame(cv::Mat& image, const cv::Mat& depth, cv::Rect &bound); //this updates dist_to_goal and angle_to_goal

   
    int _hue_min = 70;                            //60-95 is a good range for bright green
    int _hue_max = 100;
    int _sat_min =  45;
    int _sat_max = 255;
    int _val_min = 175;
    int _val_max = 255;

    /*int _hue_min = 60;                            //60-95 is a good range for bright green
    int _hue_max = 95;
    int _sat_min = 180;
    int _sat_max = 255;
    int _val_min = 67;
    int _val_max = 255;*/

	bool _draw;

private:
	void wrapConfidence(float &confidence);
	ObjectType _goal_shape;
    cv::Point2f _fov_size;
	cv::Size _frame_size;
    const float _goal_height = 2.159 - 1;  // goal height minus camera mounting ht

    float _dist_to_goal;
    float _angle_to_goal;
    bool  _goal_found;
	float _min_valid_confidence;

    bool generateThreshold(const cv::Mat& ImageIn, cv::Mat& ImageOut, int H_MIN, int H_MAX, int S_MIN, int S_MAX, int V_MIN, int V_MAX);
};
