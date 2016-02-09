#include <iostream>
#include <opencv2/core/core.hpp>
#include "track.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

ObjectType::ObjectType(int contour_type_id=1) {
	switch(contour_type_id) {
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
		case default:
			cerr << "error initializing object!" << endl;
	}

}

ObjectType::ObjectType(vector< cv::Point2f > contour_in) {
	_contour = contour_in;
}

void ObjectType::computeProperties() {
	cv::Rect br = boundingRect(_contour);
	_width = br.width();
	_height = br.height();
	_area = contourArea(_contour);
	//TODO finish this with com
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

// Adjust position based on camera motion
// between frames
void TrackedObject::setPosition(const cv::Point2f &new_position)
{
	_position = new_position;
}

// Set the distance to the bin for the current frame
void TrackedObject::setPosition(const cv::Rect &screen_position, const double avg_depth)
{
	cv::Point rect_center;
	rect_center.x = screen_position.tl().x + (screen_position.width/2);
	rect_center.y = screen_position.tl().y - (screen_position.height/2);
	cv::Point dist_to_center = rect_center - Point(_frame_size.width,_frame_size.height);
	cv::Point2f percent_fov;
	percent_fov.x = (float)dist_to_center.x / (float)_frame_size.width;
	percent_fov.y = (float)dist_to_center.y / (float)_frame_size.height;
	cv::Point2f angle = Point2f(percent_fov.x * _fov_size.width, percent_fov.y * _fov_size.height);

	_position.x = avg_depth * sin(percent_fov.x) * cos(percent_fov.y); //x=rsin(th1) * cos(th2)
	_position.y = avg_depth * sin(percent_fov.x) * sin(percent_fov.y); //y=rsin(th1) * sin(th2)
	_position.z = avg_depth * cos(percent_fov.x); //x=rcos(th1)
	
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
	float r = sqrt(_position.x * _position.x + _position.y * _position.y + _position.z * _position.z);

}

// Return the area of the tracked object
double TrackedObject::area(void) const
{
	return _position.width * _position.height;
}

// Return the position of the tracked object
cv::Rect TrackedObject::getPosition(void) const
{
	return _position;
}

// Update current object position
// Maybe maintain a range of previous positions seen +/- some margin instead?
cv::Rect TrackedObject::setPosition(const cv::Rect &position)
{
	return _position = position;
}

double TrackedObject::getAverageDistance(double &stdev) const
{
	//std::cout << "\tAverageDistance" << std::endl;
	return getAverageAndStdev(_distanceArray, stdev);
}
double TrackedObject::getAverageAngle(double &stdev) const
{
	//std::cout << "\tAverageAngle" << std::endl;
	return getAverageAndStdev(_angleArray, stdev);
}

std::string TrackedObject::getId(void) const
{
	return _id;
}

// Helper function to average distance and angle
double TrackedObject::getAverageAndStdev(double *list, double &stdev) const
{
	double sum        = 0.0;
	size_t validCount = 0;
	size_t seenCount  = 0;
	// Work backwards from _listIndex.  Find the first _dataLength valid entries and get the average
	// of those.  Make sure it doesn't loop around multiple times
	for (size_t i = _listIndex; (seenCount < _listLength) && (validCount < _dataLength); i--)
	{
		if (_detectArray[i % _listLength])
		{
			validCount += 1;
			sum += list[i % _listLength];
		}
		seenCount += 1;
	}

	// Nothing valid?  Return 0s
	if (validCount == 0)
	{
	   stdev = 0.0;
	   return 0.0;
	}

	double average   = sum / validCount;
	double sumSquare = 0.0;
	for (size_t i = _listIndex; (seenCount < _listLength) && (validCount < _dataLength); i--)
	{
		if (_detectArray[i % _listLength])
			sumSquare += (list[i % _listLength] - average) * (list[i % _listLength] - average);
		seenCount += 1;
	}
	stdev = sumSquare / validCount;

	// Code is returning NaN - test here since NaN is never equal to any
	// number including another NaN.
	if (average != average)
	   average = 0.0;
	if (stdev != stdev)
	   stdev = 0.0;
	return average;
}

// Create a tracked object list.  Set the object width in inches
// (feet, meters, parsecs, whatever) and imageWidth in pixels since
// those stay constant for the entire length of the run
TrackedObjectList::TrackedObjectList(double objectWidth, int imageWidth) :
_imageWidth(imageWidth),
_detectCount(0),
_objectWidth(objectWidth)
{
}
#if 0
void Add(const cv::Rect &position)
{
_list.push_back(TrackedObject(position));
}
#endif
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
void TrackedObjectList::adjustPosition(const cv::Mat &transformMat)
{
	for (auto it = _list.begin(); it != _list.end(); ++it)
		it->adjustPosition(transformMat);
}

// Simple printout of list into stdout
void TrackedObjectList::print(void) const
{
	for (auto it = _list.cbegin(); it != _list.cend(); ++it)
	{
		double stdev;
		double average = it->getAverageDistance(stdev);
		std::cout << it->getId() << " distance " << average << "+-" << stdev << " ";
		average = it->getAverageAngle(stdev);
		std::cout << " angle " << average << "+-" << stdev << std::endl; 
	}
}

// Return list of detect info for external processing
void TrackedObjectList::getDisplay(std::vector<TrackedObjectDisplay> &displayList) const
{
	displayList.clear();
	TrackedObjectDisplay tod;
	for (auto it = _list.cbegin(); it != _list.cend(); ++it)
	{
		double stdev;
		tod.distance = it->getAverageDistance(stdev);
		tod.angle    = it->getAverageAngle(stdev);
		tod.rect     = it->getPosition();
		tod.id       = it->getId();
		tod.ratio    = it->getDetectedRatio();
		displayList.push_back(tod);
	}
}

// Process a detected rectangle from the current frame.
// This will either match a previously detected object or
// if not, add a new object to the list
void TrackedObjectList::processDetect(const cv::Rect &detectedRect)
{
	const double areaDelta = 0.40;
	double rectArea = detectedRect.width * detectedRect.height;
	cv::Point rectCorner(detectedRect.x, detectedRect.y);
	//std::cout << "Processing " << detectedRect.x << "," << detectedRect.y << std::endl;
	for (auto it = _list.begin(); it != _list.end(); ++it)
	{
		// Look for object with roughly the same position 
		// as the current rect
		//std::cout << "\t distance " << it->distanceFromPoint(rectCorner) << std::endl;
		if( it->distanceFromPoint(rectCorner) < 2000) // tune me!
		{
			// And roughly the same area - +/- areaDelta %
			double itArea = it->area();
			//std::cout << "\t area " << (rectArea * (1.0 - areaDelta)) << "-" << (rectArea * (1.0 + areaDelta)) << " vs " <<  itArea << std::endl;
			if (((rectArea * (1.0 - areaDelta)) < itArea) &&
				((rectArea * (1.0 + 2*areaDelta)) > itArea) )
			{
				//std::cout << "\t Updating " << it->getId() << std::endl;
				it->setDistance(detectedRect, _objectWidth, _imageWidth);
				it->setAngle(detectedRect, _imageWidth);
				it->setPosition(detectedRect);
				return;
			}
		}
	}
	// Object didn't match previous hits - add a new one to the list
	TrackedObject to(detectedRect, _detectCount++);
	to.setDistance(detectedRect, _objectWidth, _imageWidth);
	to.setAngle(detectedRect, _imageWidth);
	_list.push_back(to);
	//std::cout << "\t Adding " << to.getId() << std::endl;
}

