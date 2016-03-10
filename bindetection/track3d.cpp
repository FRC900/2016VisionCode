#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "track3d.hpp"
#include "hungarian.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include <limits>

const int missedFrameCountMax = 10;

ObjectType::ObjectType(int contour_type_id=1) {
	switch(contour_type_id) {
		//loads one of the preset shapes into the

		case 1: //a ball!
			{
				float ball_diameter = 0.2476; // meters
				contour_.push_back(cv::Point2f(0,0));
				contour_.push_back(cv::Point2f(0,ball_diameter));
				contour_.push_back(cv::Point2f(ball_diameter,ball_diameter));
				contour_.push_back(cv::Point2f(ball_diameter,0));
			}
			break;

		case 2: //a bin (just because)
			contour_.push_back(cv::Point2f(0,0));
			contour_.push_back(cv::Point2f(0,0.5842));
			contour_.push_back(cv::Point2f(0.5842,0.5842));
			contour_.push_back(cv::Point2f(0.5842,0));
			break;

		case 3: //the vision goal
			{
				float max_y = .3048;
				contour_.push_back(cv::Point2f(0, max_y - 0));
				contour_.push_back(cv::Point2f(0, max_y - 0.3048));
				contour_.push_back(cv::Point2f(0.0508, max_y - 0.3048));
				contour_.push_back(cv::Point2f(0.0508, max_y - 0.0508));
				contour_.push_back(cv::Point2f(0.508-0.0508, max_y - 0.0508));
				contour_.push_back(cv::Point2f(0.508-0.0508, max_y - 0.3048));
				contour_.push_back(cv::Point2f(0.508, max_y - 0.3048));
				contour_.push_back(cv::Point2f(0.508, max_y - 0));
			}
			break;

		default:
			std::cerr << "error initializing object!" << std::endl;
	}

	computeProperties();

}

ObjectType::ObjectType(const std::vector< cv::Point2f > &contour_in) :
	contour_(contour_in)
{
	computeProperties();
}

ObjectType::ObjectType(const std::vector< cv::Point > &contour_in)
{
	for(size_t i = 0; i < contour_in.size(); i++)
	{
		cv::Point2f p;
		p.x = (float)contour_in[i].x;
		p.y = (float)contour_in[i].y;
		contour_.push_back(p);
	}
	computeProperties();

}

void ObjectType::computeProperties()
{
	float min_x = std::numeric_limits<float>::max();
	float min_y = std::numeric_limits<float>::max();
	float max_x = std::numeric_limits<float>::min();
	float max_y = std::numeric_limits<float>::min();
	for (auto it = contour_.cbegin(); it != contour_.cend(); ++it)
	{
		min_x = std::min(min_x, it->x);
		min_y = std::min(min_y, it->y);
		max_x = std::max(max_x, it->x);
		max_y = std::max(max_y, it->y);
	}
	width_ = max_x - min_x;
	height_ = max_y - min_y;
	area_ = cv::contourArea(contour_);

	//compute moments and use them to find center of mass
	cv::Moments mu = moments(contour_, false);
	com_ = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
}

bool ObjectType::operator== (const ObjectType &t1) const {
	return this->shape() == t1.shape();
}


static cv::Point3f screenToWorldCoords(const cv::Rect &screen_position, double avg_depth, const cv::Point2f &fov_size, const cv::Size &frame_size, float cameraElevation)
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

	cv::Point2f rect_center(
			screen_position.tl().x + (screen_position.width  / 2.0),
			screen_position.tl().y + (screen_position.height / 2.0));
	cv::Point2f dist_to_center(
			rect_center.x - (frame_size.width / 2.0),
			-rect_center.y + (frame_size.height / 2.0));
	cv::Point2f percent_fov(
			dist_to_center.x / frame_size.width,
			dist_to_center.y / frame_size.height);

	float azimuth = percent_fov.x * fov_size.x;
	float inclination = percent_fov.y * fov_size.y - cameraElevation;

	cv::Point3f retPt(
			avg_depth * cos(inclination) * sin(azimuth),
			avg_depth * cos(inclination) * cos(azimuth),
			avg_depth * sin(inclination));

	//std::cout << "Distance to center: " << dist_to_center << std::endl;
	//std::cout << "Actual Inclination: " << inclination << std::endl;
	//std::cout << "Actual Azimuth: " << azimuth << std::endl;
	//std::cout << "Actual location: " << retPt << std::endl;

	return retPt;
}

static cv::Rect worldToScreenCoords(const cv::Point3f &_position, ObjectType _type, const cv::Point2f &fov_size, const cv::Size &frame_size, float cameraElevation)
{
	float r = sqrtf(_position.x * _position.x + _position.y * _position.y + _position.z * _position.z) + (4.572 * 25.4)/1000.0;
	float azimuth = asin(_position.x / sqrt(_position.x * _position.x + _position.y * _position.y));
	float inclination = asin( _position.z / r ) + cameraElevation;

	cv::Point2f percent_fov = cv::Point2f(azimuth / fov_size.x, inclination / fov_size.y);
	cv::Point2f dist_to_center(percent_fov.x * frame_size.width,
			                   percent_fov.y * frame_size.height);

	cv::Point2f rect_center(
			dist_to_center.x + (frame_size.width / 2.0),
			-dist_to_center.y + (frame_size.height / 2.0));

	cv::Point2f angular_size( 2.0 * atan2(_type.width(), (2.0*r)), 2.0 * atan2(_type.height(), (2.0*r)));
	cv::Point2f screen_size(
			angular_size.x * (frame_size.width / fov_size.x),
			angular_size.y * (frame_size.height / fov_size.y));

	cv::Point topLeft(
			cvRound(rect_center.x - (screen_size.x / 2.0)),
			cvRound(rect_center.y - (screen_size.y / 2.0)));
			return cv::Rect(topLeft.x, topLeft.y, cvRound(screen_size.x), cvRound(screen_size.y));
}

TrackedObject::TrackedObject(int id,
							 const ObjectType &type_in,
							 const cv::Rect   &screen_position,
							 double            avg_depth,
							 cv::Point2f       fov_size,
							 cv::Size          frame_size,
							 float             camera_elevation,
							 float             dt,
							 float             accel_noise_mag,
							 size_t            historyLength) :
		_type(type_in),
		_detectHistory(historyLength),
		_positionHistory(historyLength),
		_KF(screenToWorldCoords(screen_position, avg_depth, fov_size, frame_size, camera_elevation),
			dt, accel_noise_mag),
		missedFrameCount_(0),
		cameraElevation_(camera_elevation)
{
	setPosition(screen_position, avg_depth, fov_size, frame_size);
	setDetected();

	// Label with base-26 letter ID (A, B, C .. Z, AA, AB, AC, etc)
	do
	{
		_id += (char)(id % 26 + 'A');
		id /= 26;
	}
	while (id != 0);
	std::reverse(_id.begin(), _id.end());
}


TrackedObject::~TrackedObject()
{
}

// Set the position based on x,y,z coords

void TrackedObject::setPosition(const cv::Point3f &new_position)
{
	_position = new_position;
	addToPositionHistory(_position);
}

// Set the position based on a rect on the screen and depth info from the zed
void TrackedObject::setPosition(const cv::Rect &screen_position, double avg_depth,
		                        const cv::Point2f &fov_size, const cv::Size &frame_size)
{
	setPosition(screenToWorldCoords(screen_position, avg_depth, fov_size, frame_size, cameraElevation_));
}

void TrackedObject::adjustPosition(const Eigen::Transform<double, 3, Eigen::Isometry> &delta_robot)
{
	//Eigen::AngleAxisd rot(0.5*M_PI, Eigen::Vector3d::UnitZ());

	Eigen::Vector3d old_pos_vec(_position.x, _position.y, _position.z);
	Eigen::Vector3d new_pos_vec = delta_robot * old_pos_vec;

	_position = cv::Point3f(new_pos_vec[0], new_pos_vec[1], new_pos_vec[2]);

	for (auto it = _positionHistory.begin(); it != _positionHistory.end(); ++it)
	{
		Eigen::Vector3d old_pos_vector(it->x, it->y, it->z);
		Eigen::Vector3d new_pos_vector = delta_robot * old_pos_vector;
		*it = cv::Point3f(new_pos_vector[0], new_pos_vector[1], new_pos_vector[2]);
	}
}

void TrackedObject::adjustPosition(const cv::Mat &transform_mat, float depth, const cv::Point2f &fov_size, const cv::Size &frame_size)
{
	//get the position of the object on the screen
	cv::Rect screen_rect = getScreenPosition(fov_size,frame_size);
	cv::Point screen_pos = cv::Point(screen_rect.tl().x + screen_rect.width / 2, screen_rect.tl().y + screen_rect.height / 2);

	//create a matrix to hold positon for matrix multiplication
	cv::Mat pos_mat(3,1,CV_64FC1);
	pos_mat.at<double>(0,0) = screen_pos.x;
	pos_mat.at<double>(0,1) = screen_pos.y;
	pos_mat.at<double>(0,2) = 1.0;

	//correct the position
	cv::Mat new_screen_pos_mat(3,1,CV_64FC1);
	new_screen_pos_mat = transform_mat * pos_mat;
	cv::Point new_screen_pos = cv::Point(new_screen_pos_mat.at<double>(0),new_screen_pos_mat.at<double>(1));

	//create a dummy bounding rect because setPosition requires a bounding rect as an input rather than a point
	cv::Rect new_screen_rect(new_screen_pos.x,new_screen_pos.y,0,0);
	setPosition(new_screen_rect,depth,fov_size,frame_size);
	//update the history
	for (auto it = _positionHistory.begin(); it != _positionHistory.end(); ++it)
	{
		screen_rect = worldToScreenCoords(*it,_type,fov_size,frame_size, cameraElevation_);
		screen_pos = cv::Point(screen_rect.tl().x + screen_rect.width / 2, screen_rect.tl().y + screen_rect.height / 2);
		pos_mat.at<double>(0,0) = screen_pos.x;
		pos_mat.at<double>(0,1) = screen_pos.y;
		pos_mat.at<double>(0,2) = 1.0;
		cv::Mat new_screen_pos_mat = transform_mat * pos_mat;
		cv::Point new_screen_pos = cv::Point(new_screen_pos_mat.at<double>(0),new_screen_pos_mat.at<double>(1));
		cv::Rect new_screen_rect(new_screen_pos.x,new_screen_pos.y,0,0);
		*it = screenToWorldCoords(new_screen_rect, depth, fov_size, frame_size, cameraElevation_);
	}

}

// Mark the object as detected in this frame
void TrackedObject::setDetected(void)
{
	_detectHistory.push_back(true);
	missedFrameCount_ = 0;
}

// Clear the object detect flag for this frame.
// Probably should only happen when moving to a new
// frame, but may be useful in other cases
void TrackedObject::clearDetected(void)
{
	_detectHistory.push_back(false);
	missedFrameCount_ += 1;
}

bool TrackedObject::tooManyMissedFrames(void) const
{
	return missedFrameCount_ > missedFrameCountMax;
}

// Keep a history of the most recent positions
// of the object in question
void TrackedObject::addToPositionHistory(const cv::Point3f &pt)
{
	_positionHistory.push_back(pt);
}

// Return the percent of last _detectHistory.size() frames
// the object was seen
double TrackedObject::getDetectedRatio(void) const
{
	int detectedCount = 0;

	// TODO : what to do for new detections?
	for (auto it = _detectHistory.begin();  it != _detectHistory.end(); ++it)
		if (*it)
			detectedCount += 1;
	double detectRatio = (double)detectedCount / _detectHistory.capacity();
	// Don't display stuff which hasn't been detected recently.
	if (missedFrameCount_ >= 4)
		detectRatio = std::min(0.01, detectRatio);
	return detectRatio;
}


cv::Rect TrackedObject::getScreenPosition(const cv::Point2f &fov_size, const cv::Size &frame_size) const
{
	return worldToScreenCoords(_position, _type, fov_size, frame_size, cameraElevation_);
}


//fit the contour of the object into the rect of it and return the area of that
//kinda gimmicky but pretty cool and might have uses in the future
double TrackedObject::contourArea(const cv::Point2f &fov_size, const cv::Size &frame_size) const
{
	cv::Rect screen_position = getScreenPosition(fov_size, frame_size);
	float scale_factor_x = (float)screen_position.width / _type.width();
	float scale_factor_y = (float)screen_position.height / _type.height();
	float scale_factor   = std::min(scale_factor_x, scale_factor_y);

	std::vector<cv::Point2f> scaled_contour;
	for(size_t i = 0; i < _type.shape().size(); i++)
	{
		scaled_contour.push_back(_type.shape()[i] * scale_factor);
	}

	return cv::contourArea(scaled_contour);
}

cv::Point3f TrackedObject::predictKF(void)
{
	return _KF.GetPrediction();
}


cv::Point3f TrackedObject::updateKF(cv::Point3f pt)
{
	return _KF.Update(pt);
}


void TrackedObject::adjustKF(const Eigen::Transform<double, 3, Eigen::Isometry> &delta_robot)
{
	_KF.adjustPrediction(delta_robot);
}

void TrackedObject::adjustKF(cv::Point3f delta_pos)
{
	_KF.adjustPrediction(delta_pos);
}


//Create a tracked object list
// those stay constant for the entire length of the run
TrackedObjectList::TrackedObjectList(const cv::Size &imageSize, const cv::Point2f &fovSize, float cameraElevation) :
	_detectCount(0),
	_imageSize(imageSize),
	_fovSize(fovSize),
	_cameraElevation(cameraElevation)
{
}

// Adjust position for camera motion between frames using fovis
void TrackedObjectList::adjustLocation(const Eigen::Transform<double, 3, Eigen::Isometry> &delta_robot)
{
	for (auto it = _list.begin(); it != _list.end(); ++it)
	{
		it->adjustPosition(delta_robot);
		it->adjustKF(delta_robot);
	}
}

// Adjust position for camera motion between frames using optical flow
void TrackedObjectList::adjustLocation(const cv::Mat &transform_mat)
{
	for (auto it = _list.begin(); it != _list.end(); ++it)
	{
		//measure the amount that the position changed and apply the same change to the kalman filter
		cv::Point3f old_pos = it->getPosition();
		//compute r and use it for depth (assume depth doesn't change)
		float r = sqrt(it->getPosition().x * it->getPosition().x + it->getPosition().y * it->getPosition().y + it->getPosition().z * it->getPosition().z);
		it->adjustPosition(transform_mat,r,_fovSize,_imageSize);
		cv::Point3f delta_pos = it->getPosition() - old_pos;

		it->adjustKF(delta_pos);
	}
}

// Simple printout of list into stdout
void TrackedObjectList::print(void) const
{
	for (auto it = _list.cbegin(); it != _list.cend(); ++it)
	{
		std::cout << it->getId() << " location ";
		cv::Point3f position = it->getPosition();
		std::cout << "(" << position.x << "," << position.y << "," << position.z << ")" << std::endl;
	}
}

// Return list of detect info for external processing
void TrackedObjectList::getDisplay(std::vector<TrackedObjectDisplay> &displayList) const
{
	displayList.clear();
	TrackedObjectDisplay tod;
	for (auto it = _list.cbegin(); it != _list.cend(); ++it)
	{
		tod.position = it->getPosition();
		tod.rect     = it->getScreenPosition(_fovSize, _imageSize);
		tod.id       = it->getId();
		tod.ratio    = it->getDetectedRatio();
		displayList.push_back(tod);
	}
}

const double dist_thresh_ = 1.0; // FIX ME!

// Process a set of detected rectangles
// Each will either match a previously detected object or
// if not, be added as new object to the list
void TrackedObjectList::processDetect(const std::vector<cv::Rect> &detectedRects,
									  const std::vector<float> depths,
									  const std::vector<ObjectType> &types)
{
	std::cout << "---------- Start of process detect --------------" << std::endl;
	print();
	std::vector<cv::Point3f> detectedPositions;
	std::cout << detectedRects.size() << " objects" << std::endl;
	for (size_t i = 0; i < detectedRects.size(); i++)
	{
		detectedPositions.push_back(
				screenToWorldCoords(detectedRects[i], depths[i], _fovSize, _imageSize, _cameraElevation));
		std::cout << "Detected rect [" << i << "] = " << detectedRects[i] << " positions[" << detectedPositions.size() - 1 << "]:" << detectedPositions[detectedPositions.size()-1] << std::endl;
	}
	// TODO :: Combine overlapping detections into one?

	// Maps tracks to the closest new detected object.
	// assignment[track] = index of closest detection
	std::vector<int> assignment;
	if (_list.size())
	{
		size_t tracks = _list.size();		          // Number of tracks
		size_t detections = detectedPositions.size(); //  number of detections

		std::vector< std::vector<double> > Cost(tracks,std::vector<double>(detections));

		// Calculate cost for each track->pair combo
		// The cost here is just the distance between them
		// Also check to see if the types are the same, if they are not then set the cost extremely high so that it's never matched
		auto it = _list.cbegin();
		for(size_t t = 0; t < tracks;  ++t, ++it)
		{
			// Point3f prediction=tracks[t]->prediction;
			// cout << prediction << endl;
			for(size_t d = 0; d < detections; d++)
			{
				const ObjectType it_type = it->getType();
				if(types[d] == it_type) {
					cv::Point3f diff = it->getPosition() - detectedPositions[d];
					Cost[t][d] = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
				} else {
					Cost[t][d] = std::numeric_limits<float>::max();
				}
			}
		}

		// Solving assignment problem (find minimum-cost assignment
		// between tracks and previously-predicted positions)
		AssignmentProblemSolver APS;
		APS.Solve(Cost, assignment, AssignmentProblemSolver::optimal);

		std::cout << "After APS : "<<std::endl;
		for(size_t i = 0; i < assignment.size(); i++)
			std::cout << i << ":" << assignment[i] << std::endl;
		// clear assignment from pairs with large distance
		for(size_t i = 0; i < assignment.size(); i++)
			if ((assignment[i] != -1) && (Cost[i][assignment[i]] > dist_thresh_))
				assignment[i] =- 1;
	}

	// Search for unassigned detects and start new tracks for them.
	// This will also handle the case where no tracks are present,
	// since assignment will be empty in that case - everything gets added
	for(size_t i = 0; i < detectedPositions.size(); i++)
	{
		if (find(assignment.begin(), assignment.end(), i) == assignment.end())
		{
			std::cout << "New assignment created " << i << std::endl;
			_list.push_back(TrackedObject(_detectCount++,types[i], detectedRects[i], depths[i], _fovSize, _imageSize));
		}
	}

	auto tr = _list.begin();
	auto as = assignment.begin();
	while ((tr != _list.end()) && (as != assignment.end()))
	{
		// If track updated less than one time, than filter state is not correct.
		std::cout << "Predict: " << std::endl;
		cv::Point3f prediction = tr->predictKF();
		std::cout << "prediction:" << prediction << std::endl;

		if(*as != -1) // If we have assigned detect, then update using its coordinates
		{
			std::cout << "Update match: " << std::endl;
			tr->setPosition(tr->updateKF(detectedPositions[*as]));
			std::cout << tr->getScreenPosition(_fovSize, _imageSize) << std::endl;
			tr->setDetected();
		}
		else          // if not continue using predictions
		{
			std::cout << "Update no match: " << std::endl;
			tr->setPosition(tr->updateKF(prediction));
			std::cout << tr->getScreenPosition(_fovSize, _imageSize) << std::endl;
			tr->clearDetected();
		}

		++tr;
		++as;
	}

	// Remove tracks which haven't been seen in a while
	for (auto it = _list.begin(); it != _list.end(); )
	{
		if (it->tooManyMissedFrames()) // For now just remove ones for
		{                              // which detectList is empty
			//std::cout << "Dropping " << it->getId() << std::endl;
			it = _list.erase(it);
		}
		else
		{
			++it;
		}
	}
	print();
	std::cout << "---------- End of process detect --------------" << std::endl;
}
