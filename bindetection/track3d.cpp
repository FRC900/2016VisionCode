#include <iostream>
#include <opencv2/core/core.hpp>
#include "track.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

ObjectType::ObjectType(int contour_type_id=1) {
	switch(contour_type_id) {
		//loads one of the preset shapes into the

		case 1: //a ball!
			_contour.push_back(cv::Point(0,0));
			_contour.push_back(cv::Point(0,0.254));
			_contour.push_back(cv::Point(0.254,0.254));
			_contour.push_back(cv::Point(0.254,0));
			break;

		case 2: //a bin (just because)
			_contour.push_back(cv::Point(0,0));
			_contour.push_back(cv::Point(0,0.5842));
			_contour.push_back(cv::Point(0.5842,0.5842));
			_contour.push_back(cv::Point(0.5842,0));
			break;

		case 3: //the vision goal
						//probably needs more code to work well but keep it in here anyways
			_contour.push_back(Point(0, 0));
			_contour.push_back(Point(0, 0.6096));
			_contour.push_back(Point(0.0508, 0.6096));
			_contour.push_back(Point(0.0508, 0.0508));
			_contour.push_back(Point(0.762, 0.0508));
			_contour.push_back(Point(0.762, 0.6096));
			_contour.push_back(Point(0.8128, 0.6096));
			_contour.push_back(Point(0.8128, 0));
			break;

		case default:
			cerr << "error initializing object!" << endl;
	}

	computeProperties();

}

ObjectType::ObjectType(vector< cv::Point2f > contour_in) {
	_contour = contour_in;

	computeProperties();
}

void ObjectType::computeProperties() {
	//create a bounding rectangle and use it to find width and height
	cv::Rect br = boundingRect(_contour);
	_width = br.width();
	_height = br.height();
	_area = contourArea(_contour);

	//compute moments and use them to find center of mass
	Moments mu = moments(_contour, false);
	_com = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
}





TrackedObject::TrackedObject(int id, ObjectType &type_in, cv::Size2f fov_size, cv::Size2f frame_size, size_t historyLength = TrackedObjectHistoryLength, size_t dataLength = TrackedObjectDataLength)
{
	_listLength    = historyLength;
	_dataLength    = dataLength;
	_detectArray   = new bool[historyLength];
	_positionArray = new cv::Point2f[historyLength];

	_fov_size = fov_size;
	_frame_size = frame_size;
	_type = type_in;

	//initiaalize detectArray to false
	for (size_t i = 0; i < historyLength; i++)
		_detectArray[i] = false;
	_listIndex     = 0;

	// Label with base-26 letter ID (A, B, C .. Z, AA, AB, AC, etc)
	do
	{
		_id += (char)(id % 26 + 'A');
		id /= 26;
	}
	while (id != 0);
	std::reverse(_id.begin(), _id.end());
}

// Copy constructor and assignement operators are needed to do a
// deep copy.  This makes new arrays for each object copied rather
// than just copy a pointer to the same array
TrackedObject::TrackedObject(const TrackedObject &object)
{
	_listLength    = object._listLength;
	_dataLength    = object._dataLength;
	_detectArray   = new bool[object._listLength];
	_positionArray = new cv::Point2f[object._listLength];
	memcpy(_detectArray, object._detectArray, sizeof(_detectArray[0]) * _listLength);
	memcpy(_positionArray, object._positionArray, sizeof(_positionArray[0]) * _listLength);
	_listIndex  = object._listIndex;
	_position   = object._position;
	_id         = object._id;
}

TrackedObject &TrackedObject::operator=(const TrackedObject &object)
{
	_listLength = object._listLength;
	_dataLength = object._dataLength;
	delete [] _detectArray;
	delete [] _distanceArray;
	delete [] _angleArray;
	_detectArray   = new bool[object._listLength];
	_distanceArray = new cv::Point2f[object._listLength];
	memcpy(_detectArray, object._detectArray, sizeof(_detectArray[0]) * _listLength);
	memcpy(_positionArray, object._positionArray, sizeof(_positionArray[0]) * _listLength);
	_listIndex  = object._listIndex;
	_position   = object._position;
	_id         = object._id;
	return *this;
}

TrackedObject::~TrackedObject()
{
	delete[] _detectArray;
	delete[] _positionArray;
}

// set position
void TrackedObject::setPosition(const cv::Point2f &new_position)
{
	_position = new_position;
}

// Set the position based on a rect on the screen and depth info from the zed
void TrackedObject::setPosition(const cv::Rect &screen_position, const double avg_depth)
{
	/*
	Method:
		find the center of the rect
		compute the distance from the center of the rect to center of image (pixels)
		convert to degrees based on fov and image size
		do a polar to cartesian cordinate conversion to find x,y,z of object
	Equations:
		x=rsin(thy) * cos(thx)
		y=rsin(thy) * sin(thx)
		z=rcos(thy)
	Notes:
		Z is up, X is left-right, and Y is forward
		0,0,0 = right in front of you
	*/

	cv::Point rect_center;
	rect_center.x = screen_position.tl().x + (screen_position.width/2);
	rect_center.y = screen_position.tl().y - (screen_position.height/2);
	cv::Point dist_to_center = rect_center - Point(_frame_size.width,_frame_size.height);
	cv::Point2f percent_fov;
	percent_fov.x = (float)dist_to_center.x / (float)_frame_size.width;
	percent_fov.y = (float)dist_to_center.y / (float)_frame_size.height;
	cv::Point2f angle = Point2f(percent_fov.x * _fov_size.width, percent_fov.y * _fov_size.height);

	_position.x = avg_depth * sin(angle.x) * cos(angle.y);
	_position.y = avg_depth * sin(angle.x) * sin(angle.y);
	_position.z = avg_depth * cos(angle.x);

}

// Mark the object as detected in this frame
void TrackedObject::setDetected(void)
{
	_detectArray[_listIndex % _listLength] = true;
}

// Clear the object detect flag for this frame.
// Probably should only happen when moving to a new
// frame, but may be useful in other cases
void TrackedObject::clearDetected(void)
{
	_detectArray[_listIndex % _listLength] = false;
}

// Return the percent of last _listLength frames
// the object was seen
double TrackedObject::getDetectedRatio(void) const
{
	int detectedCount = 0;
	int i;
	bool recentHits = true;

	// Don't display detected bins if they're not seen for at least 1 of 4 consecutive frames
	if (_listIndex > 4)
	{
		recentHits = false;
		for (i = _listIndex; (i >= 0) && (i >= (int)_listIndex - 4) && !recentHits; i--)
			if (_detectArray[i % _listLength])
				recentHits = true;
	}

	for (size_t j = 0; j < _listLength; j++)
		if (_detectArray[j])
			detectedCount += 1;
	double detectRatio = (double)detectedCount / _listLength;
	if (!recentHits)
		detectRatio = std::min(0.1, detectRatio);
	return detectRatio;
}

// Increment to the next frame
void TrackedObject::nextFrame(void)
{
	_listIndex += 1;
	clearDetected();
}


cv::Point3f TrackedObject::getPosition() const
{
	return _position;
}

cv::Rect TrackedObject::getScreenPosition() const
{
	/*
	Reversal of the setPosition method
	Could be used for things like linear interpolation or prediction of rect location
	*/
	float r = sqrt(_position.x * _position.x + _position.y * _position.y + _position.z * _position.z);
	float theta_2 = arcsin( _position.y / (r * sin(acos(_position.z / r))) );
	float theta_1 = arcsin( _position.x / (r * cos(theta_2)));

	cv::Point2f percent_fov = Point2f(theta_2 / _fov_size.x, theta_1 / _fov_size.y);
	cv::Point dist_to_center = Point(percent_fov.x * _frame_size.width, percent_fov.y * _frame_size.height);

	cv::Point rect_center;
	rect_center.x = dist_to_center.x + (_frame_size.width / 2);
	rect_center.y = dist_to_center.y + (_frame_size.height / 2);

	cv::Point2f angular_size = Point2f(2.0 * atan(_type.width() / (2*r)), 2.0 * atan(_type.height() / (2*r)));
	cv::Point2f screen_size;
	screen_size.x = angular_size.x * (_frame_size.width / _fov_size.x);
	screen_size.y = angular_size.y * (_frame_size.height / _fov_size.y);

	cv::Point topLeft;
	topLeft.x = rect_center.x - (screen_size.x / 2);
	topLeft.y = rect_center.y - (screen_size.y / 2);

	return Rect(topLeft.x, topLeft.y, screen_size.x, screen_size.y);

}

// Return the area of the boundingRect of the object
double TrackedObject::rectArea(void) const
{
	cv::Rect screen_position = getScreenPosition();
	return screen_position.width * screen_position.height;
}

//fit the contour of the object into the rect of it and return the area of that
//kinda gimmicky but pretty cool and might have uses in the future
double TrackedObject::contourArea(void) const
{
	cv::Rect screen_position = getScreenPosition();
	float scale_factor_x = screen_position.width / _type.width();
	float scale_factor_y = screen_position.height / _type.height();
	float scale_factor;

	if(scale_factor_x < scale_factor_y)
		scale_factor = scale_factor_x;
	else
		scale_factor = scale_factor_y;

	vector<Point> scaled_contour;
	for(int i = 0; i < _type.shape().size(); i++)
	{
		scaled_contour.push_back(_type.shape()[i] * scale_factor);
	}

	return cv::contourArea(scaled_contour);
}

// Helper function to average distance and angle
cv::Point3f TrackedObject::getAveragePosition(cv::Point3f &variance) const
{
	cv::Point3f sum = Point3f(0,0,0);
	cv::Point3f sumDeviation = Point3f(0,0,0);
	size_t validCount = 0;
	size_t seenCount  = 0;
	// Work backwards from _listIndex.  Find the first _dataLength valid entries and get the average
	// of those.  Make sure it doesn't loop around multiple times
	for (size_t i = _listIndex; (seenCount < _listLength) && (validCount < _dataLength); i--)
	{
		if (_detectArray[i % _listLength])
		{
			validCount += 1;
			sum.x += _positionArray[i % _listLength].x;
			sum.y += _positionArray[i % _listLength].y;
			sum.z += _positionArray[i % _listLength].z;
		}
		seenCount += 1;
	}

	// Nothing valid?  Return 0s
	if (validCount == 0)
	{
	   variance = Point3f(0,0,0);
	   return Point3f(0,0,0);
	}

	cv::Point3f average;
	average.x = sum.x / validCount;
	average.y = sum.y / validCount;
	average.z = sum.z / validCount;

	double sumSquare = 0.0;
	for (size_t i = _listIndex; (seenCount < _listLength) && (validCount < _dataLength); i--)
	{
		if (_detectArray[i % _listLength]) {
			sumDeviation.x += (list[i % _listLength].x - average.x) * (list[i % _listLength].x - average.x);
			sumDeviation.y += (list[i % _listLength].y - average.y) * (list[i % _listLength].y - average.y);
			sumDeviation.z += (list[i % _listLength].z - average.z) * (list[i % _listLength].z - average.z);
		}
		seenCount += 1;
	}
	variance = Point3f(sumDeviation.x / (validCount-1), sumDeviation.y / (validCount-1) ,sumDeviation.z / (validCount-1));

	// Code is returning NaN - test here since NaN is never equal to any
	// number including another NaN.
	if (average != average)
	   average = Point3f(0,0,0);
	if (stdev != stdev)
	   stdev = Point3f(0,0,0);
	return average;
}


cv::Point3f TrackedObject::getAveragePosition(double &variance) const
{
	//compute variance as a single number by squaring variance in
	//x,y,z and square rooting the result
	//this may or may not be a useful measure
	cv::Point3f variance_3d;
	cv::Point3f avg;
	avg = getAveragePosition(variance_3d)
	variance = sqrt(variance_3d.x * variance_3d.x + variance_3d.y * variance_3d.y + variance_3d.z * variance_3d.z);
	return avg;
}

int TrackedObject::lastSeen() {
	//loop through the list backwards and check if detected
	int last_seen_index = 0;
	for(size_t i = _listIndex; (seenCount < _listLength) && (validCount < _dataLength); i--) {
		if(_detectArray)
			break;
		last_seen_index++;
		}
	return last_seen_index;
}


//Create a tracked object list
// those stay constant for the entire length of the run
TrackedObjectList::TrackedObjectList(cv::Size imageSize, cv::Size fovSize) {

	_imageSize = imageSize;
	_fovSize = fovSize;
	_detectCount = 0;

}
// Go to the next frame.  First remove stale objects from the list
// and call nextFrame on the remaining ones
void TrackedObjectList::nextFrame(void)
{
	for (auto it = _list.begin(); it != _list.end(); )
	{
		if (it->getDetectedRatio() < 0.00001) // For now just remove ones for
		{                                     // which detectList is empty
			//std::cout << "Dropping " << it->getId() << std::endl;
			it = _list.erase(it);
		}
		else
		{
			it->nextFrame();
			++it;
		}
	}
}

// Adjust position for camera motion between frames
void TrackedObjectList::adjustLocation(const Eigen::Isometry3d &delta_robot)
{
	for (auto it = _list.begin(); it != _list.end(); ++it) {
		cv::Point3f old_position = it->getPosition();
		Eigen::Vector3d = old_pos_vector(old_position.x, old_position.y, old_position.z);
		Eigen::Vector3d new_pos_vector = delta_robot.inverse() * old_pos_vector;
		it->setPosition(Point3f(new_pos_vector.x, new_pos_vector.y, new_pos_vector.z));
	}
}

// Simple printout of list into stdout
void TrackedObjectList::print(void) const
{
	for (auto it = _list.cbegin(); it != _list.cend(); ++it)
	{
		float variance;
		cv::Point3f average = it->getAverageDistance(stdev);
		std::cout << it->getId() << " location ";
		std::cout << "(" << average.x << "," << average.y << "," << average.z << ")";
		std::cout << "+-" << variance << " " << std::endl;
	}
}

// Return list of detect info for external processing
void TrackedObjectList::getDisplay(std::vector<TrackedObjectDisplay> &displayList) const
{
	displayList.clear();
	TrackedObjectDisplay tod;
	for (auto it = _list.cbegin(); it != _list.cend(); ++it)
	{
		cv::Point3f stdev;
		tod.position = it->getPosition(stdev);
		tod.rect     = it->getScreenPosition();
		tod.id       = it->getId();
		tod.ratio    = it->getDetectedRatio();
		displayList.push_back(tod);
	}
}

// Process a detected rectangle from the current frame.
// This will either match a previously detected object or
// if not, add a new object to the list
void TrackedObjectList::processDetect(const cv::Rect &detectedRect, ObjectType type)
{
	TrackedObject new_object(_detectCount++, type, _fovSize, _imageSize);
	cv::Point3f new_object_pos = new_object.getPosition();

	for (auto it = _list.begin(); it != _list.end(); ++it)
	{
		cv::Point3f distance;
		distance.x = new_object_pos.x - it->getPosition().x;
		distance.y = new_object_pos.y - it->getPosition().y;
		distance.z = new_object_pos.z - it->getPosition().z;

		int last_seen = it->lastSeen();
		//use a linear equation for the maximum distance away that
		//the bin can be with the x parameter being the number of frames
		//since it was last seen
		float last_seen_multiplier = 0.125; // tune me!
		float initial_distance_threshold = 1.0; // tune me! (this is in m)
		if( distance.x < initial_distance_threshold + last_seen_multiplier * (float)last_seen &&
				distance.y < initial_distance_threshold + last_seen_multiplier * (float)last_seen &&
				distance.z < initial_distance_threshold + last_seen_multiplier * (float)last_seen)
		{
				it->setPosition(new_object_pos);
				return;
			}
		}
	}
	// Object didn't match previous hits - add a new one to the list
	_list.push_back(new_object);
	//std::cout << "\t Adding " << to.getId() << std::endl;
}
