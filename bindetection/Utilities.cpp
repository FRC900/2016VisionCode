#include "Utilities.hpp"

using namespace std;

namespace utils {

	std::pair<float, float> minOfDepthMat(const cv::Mat& img, const cv::Mat& mask, const cv::Rect& bound_rect, int range) {

		if (img.empty())
			return make_pair(-1,-1);
		if ((img.rows != mask.rows) || (img.cols != mask.cols))
			return make_pair(-2,-2);

		double min = numeric_limits<float>::max();
		double max = numeric_limits<float>::min();
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

	void printIsometry(const Eigen::Transform<double, 3, Eigen::Isometry> m) {

		Eigen::Vector3d xyz = m.translation();
		Eigen::Vector3d rpy = m.rotation().eulerAngles(0, 1, 2);
		cout << "Camera Translation: " << xyz << endl;
		cout << "Camera Rotation: " << rpy << endl;
		
	}

	double normalCFD(double mean, double stddev, double value)
	{
		double z_score = (value - mean) / stddev;
   		return 0.5 * erfc(-z_score * M_SQRT1_2);
	}

}
