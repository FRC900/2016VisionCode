// https://raw.githubusercontent.com/Smorodov/Multitarget-tracker/master/KalmanFilter/Kalman.cpp
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "kalman.hpp"
using namespace std;
using namespace cv;
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
TKalmanFilter::TKalmanFilter(Point3f pt,float dt,float Accel_noise_mag) :
	kalman(KalmanFilter(6, 3, 0, CV_32F)) // 6 state variables, 3 measurements
{
	cout << "Created KF" << endl;
	//time increment (lower values makes target more "massive")

	// We don't know acceleration, so, assume it to process noise.
	// But we can guess, the range of acceleration values thich can be achieved by tracked object. 
	// Process noise. (standard deviation of acceleration: �/�^2)
	// shows, how much target can accelerate.
	//float Accel_noise_mag = 0.5; 

	// Transition matrix
	// x' = x + v_x*dt
	// y' = y + v_y*dt
	// z' = z + v_z*dt - 1/2gdt^2
	// v_x' = v_x
	// v_y' = v_y
	// v_z' = v_z - gdt
	kalman.transitionMatrix = (Mat_<float>(6, 6) << 
			1,0,0, dt, 0,  0,   
			0,1,0, 0,  dt, 0, 
			0,0,1, 0,  0,  dt,  
			0,0,0, 0.5,  0,  0,  
			0,0,0, 0,  0.5,  0,
			0,0,0, 0,  0,  0.5);

#if 0
	// Add gravity to z position and velocity.  Probably
	// best not to use this since we're mostly looking
	// for targets on the ground rather than in flight
	kalman.controlMatrix = (Mat_<float>(6,1) << 
			0,
			0,
			-0.5*9.81*dt*dt,
			0,
			0,
			-9.81*dt);
#endif
	// init... 
	kalman.statePre.at<float>(0) = pt.x; // x
	kalman.statePre.at<float>(1) = pt.y; // y
	kalman.statePre.at<float>(2) = pt.z; // z
	kalman.statePre.at<float>(4) = 0;    // v_x
	kalman.statePre.at<float>(5) = 0;    // v_y
	kalman.statePre.at<float>(6) = 0;    // v_z

	kalman.statePost = kalman.statePre;

	setIdentity(kalman.measurementMatrix);

	kalman.processNoiseCov=(Mat_<float>(6, 6) << 
			pow(dt,4.0)/4.0, 0,				   0,               pow(dt,3.0)/2.0, 0,               0,
			0,               pow(dt,4.0)/4.0,  0,               0,               pow(dt,3.0)/2.0, 0,
			0,			     0,				   pow(dt,4.0)/4.0, 0,               0,               pow(dt,3.0)/2.0,
			pow(dt,3.0)/2.0, 0,				   0,               pow(dt,2.0),     0,               0,
			0,				 pow(dt,3.0)/2.0,  0,               0,               pow(dt,2.0),     0,
			0,				 0,                pow(dt,3.0)/2.0, 0,               0,               pow(dt,2.0));

	kalman.processNoiseCov *= Accel_noise_mag;

	setIdentity(kalman.measurementNoiseCov, Scalar::all(0.1));

	setIdentity(kalman.errorCovPost, Scalar::all(.1));

	cout << "transitionMatrix : " << endl << kalman.transitionMatrix << endl;
	cout << "controlMatrix : " << endl << kalman.controlMatrix << endl;
	cout << "errorCovPre : " << endl << kalman.errorCovPre << endl;
	cout << "errorCovPost : " << endl << kalman.errorCovPost << endl;
	cout << "gain : " << endl << kalman.gain << endl;
	cout << "measurement: " << endl << kalman.measurementMatrix << endl;
	cout << "measurementNoiseCov: " << endl << kalman.measurementNoiseCov << endl;
	cout << "processNoiseCov: " << endl << kalman.processNoiseCov << endl;
	cout << "statePre " << endl << kalman.statePre << endl;
	cout << "statePost " << endl << kalman.statePost << endl;
}

//---------------------------------------------------------------------------
TKalmanFilter::~TKalmanFilter()
{
}

//---------------------------------------------------------------------------
Point3f TKalmanFilter::GetPrediction()
{
	Mat prediction = kalman.predict();
	cout <<  "Predict, state out : " << kalman.statePost << endl;
	return Point3f(prediction.at<float>(0),prediction.at<float>(1),prediction.at<float>(2)); 
}
//---------------------------------------------------------------------------
Point3f TKalmanFilter::Update(Point3f p)
{
	Mat measurement(3, 1, CV_32F);
	measurement.at<float>(0) = p.x;  //update using measurements
	measurement.at<float>(1) = p.y;
	measurement.at<float>(2) = p.z;

	// Correction
	Mat estimated = kalman.correct(measurement);
	return Point3f(estimated.at<float>(0), estimated.at<float>(1), estimated.at<float>(2));
}
//---------------------------------------------------------------------------
