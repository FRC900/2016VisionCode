//standard include
#include <math.h>

//opencv include
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

//fovis include
#include "zedin.hpp"

#include <fovis.hpp>

//zed include
#include <zed/Mat.hpp>
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>

class FovisLocalizer {

public:

	FovisLocalizer(sl::zed::CamParameters input_params,int in_width, int in_height, cv::Mat& initial_frame);

	void processFrame(cv::Mat& img, cv::Mat& depth);
	std::pair<cv::Vec3f,cv::Vec3f> getTransform() const { return _transform; }
	void reloadFovis();

	int fv_param_max_pyr_level = 3;
	int fv_param_feature_search_window = 25;
	int fv_param_feature_window_size = 9; //fovis parameters
	int fv_param_target_ppf = 250;

	int num_optical_flow_sectors_x = 8;
	int num_optical_flow_sectors_y = 6; //optical flow parameters
	int num_optical_flow_points = 2000;
	int flow_arbitrary_outlier_threshold_int = 500;

private:

	std::pair<cv::Vec3f,cv::Vec3f> _transform;

	fovis::CameraIntrinsicsParameters _rgb_params;
	fovis::Rectification* _rect;
	fovis::VisualOdometry* _odom;

	cv::Mat frame, frameGray, prev, prevGray, depthFrame;

	int _im_height;
	int _im_width;

};
