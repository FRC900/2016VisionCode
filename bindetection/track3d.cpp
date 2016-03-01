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

const int missedFrameCountMax = 5;

ObjectType::ObjectType(int contour_type_id=1) {
	switch(contour_type_id) {
		//loads one of the preset shapes into the

		case 1: //a ball!
			{
				float ball_diameter = 0.2476; // meters
				_contour.push_back(cv::Point2f(0,0));
				_contour.push_back(cv::Point2f(0,ball_diameter));
				_contour.push_back(cv::Point2f(ball_diameter,ball_diameter));
				_contour.push_back(cv::Point2f(ball_diameter,0));
			}
			break;

		case 2: //a bin (just because)
			_contour.push_back(cv::Point2f(0,0));
			_contour.push_back(cv::Point2f(0,0.5842));
			_contour.push_back(cv::Point2f(0.5842,0.5842));
			_contour.push_back(cv::Point2f(0.5842,0));
			break;

		case 3: //the vision goal
			{
				float max_y = .3048;
				_contour.push_back(cv::Point2f(0, max_y - 0));
				_contour.push_back(cv::Point2f(0, max_y - 0.3048));
				_contour.push_back(cv::Point2f(0.0508, max_y - 0.3048));
				_contour.push_back(cv::Point2f(0.0508, max_y - 0.0508));
				_contour.push_back(cv::Point2f(0.508-0.0508, max_y - 0.0508));
				_contour.push_back(cv::Point2f(0.508-0.0508, max_y - 0.3048));
				_contour.push_back(cv::Point2f(0.508, max_y - 0.3048));
				_contour.push_back(cv::Point2f(0.508, max_y - 0));
			}
			break;

		default:
			std::cerr << "error initializing object!" << std::endl;
	}

	computeProperties();

}

ObjectType::ObjectType(const std::vector< cv::Point2f > &contour_in) :
	_contour(contour_in)
{
	computeProperties();
}

ObjectType::ObjectType(const std::vector< cv::Point > &contour_in) {

for(size_t i = 0; i < contour_in.size(); i++) {
	cv::Point2f p;
	p.x = (float)contour_in[i].x;
	p.y = (float)contour_in[i].y;
	_contour.push_back(p);
}
computeProperties();

}

void ObjectType::computeProperties() {
	float min_x = std::numeric_limits<float>::max();
	float min_y = std::numeric_limits<float>::max();
	float max_x = std::numeric_limits<float>::min();
	float max_y = std::numeric_limits<float>::min();
	for (auto it = _contour.cbegin(); it != _contour.cend(); ++it)
	{
		min_x = std::min(min_x, it->x);
		min_y = std::min(min_y, it->y);
		max_x = std::max(max_x, it->x);
		max_y = std::max(max_y, it->y);
	}
	_width = max_x - min_x;
	_height = max_y - min_y;
	_area = cv::contourArea(_contour);

	//compute moments and use them to find center of mass
	cv::Moments mu = moments(_contour, false);
	_com = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
}


static cv::Point3f screenToWorldCoords(const cv::Rect &screen_position, double avg_depth, const cv::Point2f &fov_size, const cv::Size &frame_size) 
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

	cv::Point2f rect_center;
	rect_center.x = screen_position.tl().x + (screen_position.width / 2.0);
	rect_center.y = screen_position.tl().y + (screen_position.height / 2.0);
	cv::Point2f dist_to_center;
	dist_to_center.x = rect_center.x - (frame_size.width / 2.0);
	dist_to_center.y = -rect_center.y + (frame_size.height / 2.0);
	//std::cout << "Distance to center: " << dist_to_center << std::endl; 
	
	cv::Point2f percent_fov;
	percent_fov.x = (float)dist_to_center.x / (float)frame_size.width;
	percent_fov.y = (float)dist_to_center.y / (float)frame_size.height;
	float azimuth = percent_fov.x * fov_size.x;
	float inclination = percent_fov.y * fov_size.y;
	
	//std::cout << "Actual Inclination: " << inclination << std::endl;
	//std::cout << "Actual Azimuth: " << azimuth << std::endl;

	cv::Point3f retPt;
	retPt.x = avg_depth * cos(inclination) * sin(azimuth);
	retPt.y = avg_depth * cos(inclination) * cos(azimuth);
	retPt.z = avg_depth * sin(inclination);
	//std::cout << "Actual location: " << retPt << std::endl;
	return retPt;
}


TrackedObject::TrackedObject( int id,
    const ObjectType &type_in,
	const cv::Rect &screen_position,
	double avg_depth,
    cv::Point2f fov_size,
    cv::Size    frame_size,
	float       dt,
	float       accel_noise_mag,
    size_t historyLength) :
	_type(type_in),
	_historyIndex(0),
	_detectHistory(std::vector<bool>(historyLength, false)),
	_KF(screenToWorldCoords(screen_position, avg_depth, fov_size, frame_size), 
        dt, accel_noise_mag),
	missedFrameCount_(0),
	positionHistoryMax_(historyLength)
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
	setPosition(screenToWorldCoords(screen_position, avg_depth, fov_size, frame_size));
}

void TrackedObject::adjustPosition(const Eigen::Transform<double, 3, Eigen::Isometry> &delta_robot)
{
	//Eigen::AngleAxisd rot(0.5*M_PI, Eigen::Vector3d::UnitZ());

	Eigen::Vector3d old_pos_vec(_position.x, _position.y, _position.z);
	Eigen::Vector3d new_pos_vec = delta_robot.inverse() * old_pos_vec;
	std::cout << "Rotation: " << delta_robot.rotation().eulerAngles(0,1,2) << std::endl;
	std::cout << "Old: " << old_pos_vec << std::endl;
	std::cout << "New: " << new_pos_vec << std::endl;
	
	//float r_old = sqrtf(_position.x * _position.x + _position.y * _position.y + _position.z * _position.z);
	//float azimuth_old = acos(_position.x / sqrtf(_position.x * _position.x + _position.y * _position.y));
	//float inclination_old = asin( _position.z / r_old );

	_position = cv::Point3f(new_pos_vec[0], new_pos_vec[1], new_pos_vec[2]);

	//float r_new = sqrtf(_position.x * _position.x + _position.y * _position.y + _position.z * _position.z);	
	//float azimuth_new = acos(_position.x / sqrtf(_position.x * _position.x + _position.y * _position.y));
	//float inclination_new = asin( _position.z / r_new );

	//std::cout << "Change in inclination: " << inclination_new - inclination_old << std::endl;
	//std::cout << "Change in azimuth: " << azimuth_new - azimuth_old << std::endl;

	for (auto it = _positionHistory.begin(); it != _positionHistory.end(); ++it) 
	{
		Eigen::Vector3d old_pos_vector(it->x, it->y, it->z);
		Eigen::Vector3d new_pos_vector = delta_robot.inverse() * old_pos_vector;
		*it = cv::Point3f(new_pos_vector[0], new_pos_vector[1], new_pos_vector[2]);
	}
}

// Mark the object as detected in this frame
void TrackedObject::setDetected(void)
{
	_detectHistory[_historyIndex % _detectHistory.size()] = true;
	missedFrameCount_ = 0;
}

// Clear the object detect flag for this frame.
// Probably should only happen when moving to a new
// frame, but may be useful in other cases
void TrackedObject::clearDetected(void)
{
	_detectHistory[_historyIndex % _detectHistory.size()] = false;
	missedFrameCount_ += 1;
}

bool TrackedObject::tooManyMissedFrames(void) const
{
	return missedFrameCount_ > missedFrameCountMax;
}

void TrackedObject::addToPositionHistory(const cv::Point3f &pt)
{
	if (_positionHistory.size() > positionHistoryMax_)
	{
		_positionHistory.erase(_positionHistory.begin(),_positionHistory.end() - positionHistoryMax_);
	}

	_positionHistory.push_back(pt);
}

// Return the percent of last _detectHistory.size() frames
// the object was seen
double TrackedObject::getDetectedRatio(void) const
{
	int detectedCount = 0;
	int i;
	bool recentHits = true;

	// Don't display detected bins if they're not seen for at least 1 of 4 consecutive frames
	if (_historyIndex > 4)
	{
		recentHits = false;
		for (i = _historyIndex; (i >= 0) && (i >= (int)_historyIndex - 4) && !recentHits; i--)
			if (_detectHistory[i % _detectHistory.size()])
				recentHits = true;
	}

	for (size_t j = 0; j < _detectHistory.size(); j++)
		if (_detectHistory[j])
			detectedCount += 1;
	double detectRatio = (double)detectedCount / _detectHistory.size();
	if (!recentHits)
		detectRatio = std::min(0.1, detectRatio);
	return detectRatio;
}

// Increment to the next frame
void TrackedObject::nextFrame(void)
{
	_historyIndex += 1;
}


cv::Rect TrackedObject::getScreenPosition(const cv::Point2f &fov_size, const cv::Size &frame_size) const 
{
	float r = sqrtf(_position.x * _position.x + _position.y * _position.y + _position.z * _position.z) + (4.572 * 25.4)/1000.0;
	//std::cout << "Position: " << _position << std::endl;
	float azimuth = asin(_position.x / sqrt(_position.x * _position.x + _position.y * _position.y));
	float inclination = asin( _position.z / r );
	//std::cout << "Computed Azimuth: " << azimuth << std::endl;
	//std::cout << "Computed Inclination: " << inclination << std::endl;
	
	cv::Point2f percent_fov = cv::Point2f(azimuth / fov_size.x, inclination / fov_size.y);
	//std::cout << "Computed Percent fov: " << percent_fov << std::endl;
	cv::Point2f dist_to_center(percent_fov.x * frame_size.width, 
			                   percent_fov.y * frame_size.height);

	cv::Point2f rect_center;
	rect_center.x = dist_to_center.x + (frame_size.width / 2.0);
	rect_center.y = -dist_to_center.y + (frame_size.height / 2.0);

	cv::Point2f angular_size = cv::Point2f( 2.0 * atan2(_type.width(), (2.0*r)), 2.0 * atan2(_type.height(), (2.0*r)));
	cv::Point2f screen_size;
	screen_size.x = angular_size.x * (frame_size.width / fov_size.x);
	screen_size.y = angular_size.y * (frame_size.height / fov_size.y);

	cv::Point topLeft;
	topLeft.x = cvRound(rect_center.x - (screen_size.x / 2.0));
	topLeft.y = cvRound(rect_center.y - (screen_size.y / 2.0));

	return cv::Rect(topLeft.x, topLeft.y, cvRound(screen_size.x), cvRound(screen_size.y));
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


//Create a tracked object list
// those stay constant for the entire length of the run
TrackedObjectList::TrackedObjectList(const cv::Size &imageSize, const cv::Point2f &fovSize) :
	_detectCount(0),
	_imageSize(imageSize),
	_fovSize(fovSize)
{
}

// Adjust position for camera motion between frames
void TrackedObjectList::adjustLocation(const Eigen::Transform<double, 3, Eigen::Isometry> &delta_robot)
{
	for (auto it = _list.begin(); it != _list.end(); ++it)
	{
		it->adjustPosition(delta_robot);
		it->adjustKF(delta_robot);
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

	for (size_t i = 0; i < detectedRects.size(); i++)
	{
		detectedPositions.push_back(
				screenToWorldCoords(detectedRects[i], depths[i], _fovSize, _imageSize));
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
		auto it = _list.cbegin();
		for(size_t t = 0; t < tracks;  ++t, ++it)
		{	
			// Point3f prediction=tracks[t]->prediction;
			// cout << prediction << endl;
			for(size_t d = 0; d < detections; d++)
			{
				cv::Point3f diff = it->getPosition() - detectedPositions[d];
				Cost[t][d] = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
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

		tr->nextFrame();

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
