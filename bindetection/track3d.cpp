#include <iostream>
#include <opencv2/core/core.hpp>
#include "track3d.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

ObjectType::ObjectType(int contour_type_id) {
	switch(contour_type_id) {
		case 1: //a ball!
			_contour.push_back(cv::Point(0,0));
			_contour.push_back(cv::Point(0,0.254));
			_contour.push_back(cv::Point(0.254,0.254));
			_contour.push_back(cv::Point(0.254,0));
			_delta_rotate = false;
			break;
		case 2: //a bin (just because)
			_contour.push_back(cv::Point(0,0));
			_contour.push_back(cv::Point(0,0.5842));
			_contour.push_back(cv::Point(0.5842,0.5842));
			_contour.push_back(cv::Point(0.5842,0));
			_delta_rotate = false;
			break;
		case default:
			cerr << "error initializing object!" << endl;
	}

}

ObjectType::ObjectType(vector< cv::Point2f > contour_in, bool delta_rotate_in=false) {
	_contour = contour_in;
	_delta_rotate = delta_rotate_in;
}

void computeProperties() {
	cv::Rect br = boundingRect(_contour);
	_width = br.width();
	_height = br.height();
	_area = contourArea(_contour);
	_com =
}

TrackedObject::TrackedObject(const cv::Rect &position, int id, size_t historyLength, size_t dataLength)
{
	_listLength    = historyLength;
	_dataLength    = dataLength;
	_detectArray   = new bool[historyLength];
	_distanceArray = new double[historyLength];
	_angleArray    = new double[historyLength];
	for (size_t i = 0; i < historyLength; i++)
		_detectArray[i] = false;
	_listIndex     = 0;
	_position      = position;
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
	_distanceArray = new double[object._listLength];
	_angleArray    = new double[object._listLength];
	memcpy(_detectArray, object._detectArray, sizeof(_detectArray[0]) * _listLength);
	memcpy(_distanceArray, object._distanceArray, sizeof(_distanceArray[0]) * _listLength);
	memcpy(_angleArray, object._angleArray, sizeof(_angleArray[0]) * _listLength);
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
	_distanceArray = new double[object._listLength];
	_angleArray    = new double[object._listLength];
	memcpy(_detectArray, object._detectArray, sizeof(_detectArray[0]) * _listLength);
	memcpy(_distanceArray, object._distanceArray, sizeof(_distanceArray[0]) * _listLength);
	memcpy(_angleArray, object._angleArray, sizeof(_angleArray[0]) * _listLength);
	_listIndex  = object._listIndex;
	_position   = object._position;
	_id         = object._id;
	return *this;
}

TrackedObject::~TrackedObject()
{
	delete[] _detectArray;
	delete[] _distanceArray;
	delete[] _angleArray;
}

// Adjust the position and angle history by
// the specified amount. Used to compensate for
// the robot turning
// TODO : also need a similar adjustTranslation call
void TrackedObject::adjustAngle(double deltaAngle, int imageWidth)
{
	// Need to figure out what positive and negative
	// angles mean
	for (size_t i = 0; i < _listLength; i++)
		_angleArray[i] += deltaAngle;
	_position.x += deltaAngle * imageWidth / HFOV;
}

// Set the distance to the bin for the current frame
void TrackedObject::setDistance(double distance)
{
	_distanceArray[_listIndex % _listLength] = distance;
	//std::cout << "\t " << getId() << " distance[" << (_listIndex & _listLength) << "] = " << distance <<std::endl;
	setDetected();
}

// Set the distance to the target using the detected
// rectangle plus the known size of the object and frame size
void TrackedObject::setDistance(const cv::Rect &rect, double objWidth, int imageWidth)
{
	double FOVFrac  = (double)rect.width / imageWidth;
	double totalFOV = (objWidth / 2.0) / FOVFrac;
	double FOVRad   =  (M_PI / 180.0) * (HFOV / 2.0);
	setDistance( (totalFOV / tan(FOVRad)) + 2.89);
}

// Set the angle off center for the current frame
void TrackedObject::setAngle(double angle)
{
	_angleArray[_listIndex % _listLength] = angle;
	//std::cout << "\t " << getId() << " angle[" << (_listIndex & _listLength) << "] = " << angle <<std::endl;
	setDetected();
}

// Set the angle off center for the current frame
// using the detected rectangle.
void TrackedObject::setAngle(const cv::Rect &rect, int imageWidth)
{
	double degreesPerPixel = HFOV / imageWidth;
	int rectCenterX = rect.x + rect.width / 2;
	int rectLocX = rectCenterX - imageWidth / 2;
	setAngle((double)rectLocX * degreesPerPixel);
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
	size_t i;
	bool recentHits = true;

	// Don't display detected bins if they're not seen for at least 1 of 4 consecutive frames
	if (_listIndex > 4)
	{
		recentHits = false;
		for (i = _listIndex; (i >= 0) && (i >= _listIndex - 4) && !recentHits; i--)
			if (_detectArray[i % _listLength])
				recentHits = true;
	}

	for (i = 0; i < _listLength; i++)
		if (_detectArray[i])
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

// Return the distance in pixels between the
// tracked object's position and a point
double TrackedObject::distanceFromPoint(cv::Point point) const
{
	//std::cout <<"position " << _position.x << "," << _position.y << " point " << point.x<<"," << point.y << std::endl;
	return (_position.x - point.x) * (_position.x - point.x) +
		   (_position.y - point.y) * (_position.y - point.y);
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

// Adjust the angle of each tracked object based on
// the rotation of the robot
// TODO : add an adjustTranslation here as well
void TrackedObjectList::adjustAngle(double deltaAngle)
{
	for (auto it = _list.begin(); it != _list.end(); ++it)
		it->adjustAngle(deltaAngle, _imageWidth);
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
