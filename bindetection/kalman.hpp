// From : https://raw.githubusercontent.com/Smorodov/Multitarget-tracker/master/KalmanFilter/Kalman.h
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
// http://www.morethantechnical.com/2011/06/17/simple-kalman-filter-for-tracking-using-opencv-2-2-w-code/
class TKalmanFilter
{
	public:
		TKalmanFilter(cv::Point3f p, float dt = 0.2, float Accel_noise_mag = 0.5);
		~TKalmanFilter();
		cv::Point3f GetPrediction();
		cv::Point3f Update(cv::Point3f p);

	private:
		cv::KalmanFilter kalman;
};

