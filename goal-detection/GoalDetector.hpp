//standard include
#include <math.h>
#include <iostream>

//opencv include
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include "Utilities.hpp"
#include "track3d.hpp"

class GoalDetector 
{
	public:

		GoalDetector(cv::Point2f fov_size, cv::Size frame_size, bool gui = false);

		cv::Rect goal_rect(void) const;
		float dist_to_goal(void) const;
		float angle_to_goal(void) const;

		// Get and set drawing flags
		void draw(bool drawFlag);
		bool draw(void) const;

		void processFrame(cv::Mat& image, const cv::Mat& depth); //this updates dist_to_goal, angle_to_goal, and _goal_rect

	private:
		void wrapConfidence(float &confidence);
		ObjectType _goal_shape;
		cv::Point2f _fov_size;
		cv::Size _frame_size;
		const float _goal_height = 2.159 - 1 - (17 * 2.54) / 100.;  // goal height minus camera mounting ht minus chopping off 17 inches.  TODO : remeasure me!

		bool  _draw;

		bool  _goal_found;
		float _dist_to_goal;
		float _angle_to_goal;
		cv::Rect _goal_rect;

		float _min_valid_confidence;

		int  _use_add_subtract;
		int  _blue_scale;
		int  _red_scale;

		int _hue_min;
		int _hue_max;
		int _sat_min;
		int _sat_max;
		int _val_min;
		int _val_max;

		bool generateThreshold(const cv::Mat& imageIn, cv::Mat& imageOut);
		bool generateThresholdAddSubtract(const cv::Mat& imageIn, cv::Mat& imageOut);
};
