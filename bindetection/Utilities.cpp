#include "Utilities.hpp"

using namespace std;
using namespace cv;

namespace utils {

	Point3f screenToWorldCoords(const cv::Point &screen_position, double avg_depth, const cv::Point2f &fov_size, const Size &frame_size, float cameraElevation)
	{
		/*
		Method:
			find the center of the rect
			compute the distance from the center of the rect to center of image (pixels)
			convert to degrees based on fov and image size
			do a polar to cartesian cordinate conversion to find x,y,z of object
		Equations:
			x=rsin(inclination) * cos(azimuth)
			y=rsin(inclination) * sin(azimuth)
			z=rcos(inclination)
		Notes:
			Z is up, X is left-right, and Y is forward
			(0,0,0) = (r,0,0) = right in front of you
		*/

		Point2f dist_to_center(
				screen_position.x - (frame_size.width / 2.0),
				-screen_position.y + (frame_size.height / 2.0));
		Point2f percent_fov(
				dist_to_center.x / frame_size.width,
				dist_to_center.y / frame_size.height);

		float azimuth = percent_fov.x * fov_size.x;
		float inclination = percent_fov.y * fov_size.y - cameraElevation;

		Point3f retPt(
				avg_depth * cosf(inclination) * sinf(azimuth),
				avg_depth * cosf(inclination) * cosf(azimuth),
				avg_depth * sinf(inclination));

		//cout << "Distance to center: " << dist_to_center << endl;
		//cout << "Actual Inclination: " << inclination << endl;
		//cout << "Actual Azimuth: " << azimuth << endl;
		//cout << "Actual location: " << retPt << endl;

		return retPt;
	}

	double slope_list(const std::vector<double>& x, const std::vector<double>& y) {
	    const auto n    = x.size();
	    const auto s_x  = std::accumulate(x.begin(), x.end(), 0.0);
	    const auto s_y  = std::accumulate(y.begin(), y.end(), 0.0);
	    const auto s_xx = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
	    const auto s_xy = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
	    const auto a    = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);
	    return a;
	}

	std::pair<float, float> minOfDepthMat(const cv::Mat& img, const cv::Mat& mask, const cv::Rect& bound_rect, int range) {

		if (img.empty())
			return make_pair(-1,-1);
		if ((img.rows != mask.rows) || (img.cols != mask.cols))
			return make_pair(-2,-2);

		float min = numeric_limits<float>::max();
		float max = numeric_limits<float>::min();
		int min_loc_x;
		int min_loc_y;
		int max_loc_x;
		int max_loc_y;
		bool found = false;
		for (int j = bound_rect.tl().y; j <= bound_rect.br().y; j++) //for each row
		{
			const float *ptr_img  = img.ptr<float>(j);
			const uchar *ptr_mask = mask.ptr<uchar>(j);

			for (int i = bound_rect.tl().x; i <= bound_rect.br().x; i++) //for each pixel in row
			{
				if ((ptr_mask[i] == 255) && !(isnan(ptr_img[i]) || (ptr_img[i] <= 0)))
				{
					found = true;
					if (ptr_img[i] > max)
					{
						max = ptr_img[i];
						max_loc_x = i;
						max_loc_y = j;
					}

					if (ptr_img[i] < min)
					{
						min = ptr_img[i];
						min_loc_x = i;
						min_loc_y = j;
					}
				}
			}
		}
		if(!found)
		{
			return make_pair(-3, -3);
		}
		float sum_min   = 0;
		int num_pix_min = 0;
		for (int j = (min_loc_y - range); j < (min_loc_y + range); j++)
		{
			const float *ptr_img  = img.ptr<float>(j);
			const uchar *ptr_mask = mask.ptr<uchar>(j);
		    for (int i = (min_loc_x - range); i < (min_loc_x + range); i++)
		    {
		        if ((0 < i) && (i < img.cols) && (0 < j) && (j < img.rows) && (ptr_mask[i] == 255) && !(isnan(ptr_img[i]) || (ptr_img[i] <= 0)))
		        {
		            sum_min += ptr_img[i];
		            num_pix_min++;
		        }
		    }
		}
		float sum_max = 0;
		int num_pix_max = 0;
		for (int j = (max_loc_y - range); j < (max_loc_y + range); j++)
		{
			const float *ptr_img  = img.ptr<float>(j);
			const uchar *ptr_mask = mask.ptr<uchar>(j);
		    for (int i = (max_loc_x - range); i < (max_loc_x + range); i++)
		    {
		        if ((0 < i) && (i < img.cols) && (0 < j) && (j < img.rows) && (ptr_mask[i] == 255) && !(isnan(ptr_img[i]) || (ptr_img[i] <= 0)))
		        {
		            sum_max += ptr_img[i];
		            num_pix_max++;
		        }
		    }
		}
		return std::make_pair(sum_min / (num_pix_min * 1000.), sum_max / (num_pix_max * 1000.));
	}

	void shrinkRect(cv::Rect &rect_in, float shrink_factor) {

		rect_in.tl() = rect_in.tl() + cv::Point(shrink_factor/2.0 * rect_in.width, shrink_factor/2.0 * rect_in.height);
		rect_in.br() = rect_in.br() - cv::Point(shrink_factor/2.0 * rect_in.width, shrink_factor/2.0 * rect_in.height);

	}
#if 0 //disabled due to Eigen issues with new JetPack version

	void printIsometry(const Eigen::Transform<double, 3, Eigen::Isometry> m) {

		Eigen::Vector3d xyz = m.translation();
		Eigen::Vector3d rpy = m.rotation().eulerAngles(0, 1, 2);
		cout << "Camera Translation: " << xyz << endl;
		cout << "Camera Rotation: " << rpy << endl;

	}
#endif

	//gets the slope that the masked area is facing away from the camera
	//useful when used with a contour to find the angle that an object is faciing
	std::pair<double,double> slopeOfMasked(const cv::Mat &depth, const cv::Mat &mask, cv::Point2f fov) {

		vector<double> slope_x_values;
		vector<double> slope_y_values;
		vector<double> slope_z_values;

		for (size_t j = 0; j < depth.rows; j++) {

			const float *ptr_depth = depth.ptr<float>(j);
			const uchar *ptr_mask = mask.ptr<uchar>(j);

			for (size_t i = 0; i < depth.cols; i++) {
				if(ptr_mask[i] == 255 && ptr_depth[i] > 0) {
					Point3f pos = screenToWorldCoords(Point(i,j), ptr_depth[i], fov, depth.size(), 0);
					slope_x_values.push_back(pos.x);
					slope_y_values.push_back(pos.y);
					slope_z_values.push_back(pos.z);
				}
			}
		}
			
		if(slope_x_values.size() <= 5)
			return std::make_pair<double,double>(0,0);
		double slope_h =  slope_list(slope_x_values, slope_y_values);
		double slope_v =  slope_list(slope_z_values, slope_y_values);

		return std::make_pair(slope_h, slope_v);

	}

	double normalCFD(const pair<double,double> &meanAndStddev, double value)
	{
		double z_score = (value - meanAndStddev.first) / meanAndStddev.second;
   		return 0.5 * erfc(-z_score * M_SQRT1_2);
	}


}
