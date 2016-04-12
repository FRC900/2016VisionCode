#include <iostream>
#include <limits>

#include "track3d.hpp"
#include "hungarian.hpp"

#include "Utilities.hpp"

using namespace std;
using namespace cv;
using namespace utils;

const int missedFrameCountMax = 10;

ObjectType::ObjectType(int contour_type_id=1) {
	switch(contour_type_id) {
		//loads one of the preset shapes into the

		case 1: //a ball!
			{
				int ball_diameter = 248; // mm
				contour_.push_back(Point(0,0));
				contour_.push_back(Point(0,ball_diameter));
				contour_.push_back(Point(ball_diameter,ball_diameter));
				contour_.push_back(Point(ball_diameter,0));
			}
			break;

		case 2: { //a bin (just because)
				contour_.push_back(Point(0,0));
				contour_.push_back(Point(0,584));
				contour_.push_back(Point(584,584));
				contour_.push_back(Point(584,0));
				break;
			}
		case 3: //the vision goal
			{
				vector<Point2f> contour_f;
				float max_y = .3048;
				contour_f.push_back(Point2f(0, max_y - 0));
				contour_f.push_back(Point2f(0, max_y - 0.3048));
				contour_f.push_back(Point2f(0.0508, max_y - 0.3048));
				contour_f.push_back(Point2f(0.0508, max_y - 0.0508));
				contour_f.push_back(Point2f(0.508-0.0508, max_y - 0.0508));
				contour_f.push_back(Point2f(0.508-0.0508, max_y - 0.3048));
				contour_f.push_back(Point2f(0.508, max_y - 0.3048));
				contour_f.push_back(Point2f(0.508, max_y - 0));

				//m to mm
				for(size_t i = 0; i < contour_f.size(); i++)
					contour_.push_back(Point(contour_f[i].x * 1000, contour_f[i].y * 1000));
				break;
			}

		default:
				cerr << "error initializing object!" << endl;
	}
	computeProperties();

}

ObjectType::ObjectType(const vector< Point > &contour_in)
{
	contour_ = contour_in;
	computeProperties();
}


//takes a rect as an input and then transforms the contour_
//to fit perfectly inside the rect
vector<Point> ObjectType::transformContour(RotatedRect roi) {
	Point2f src_points[4];
	Point2f dst_points[4];

	RotatedRect contour_rect = minAreaRect(contour_);
	contour_rect.points(src_points);

	roi.points(dst_points);

	Mat transform = getPerspectiveTransform(src_points,dst_points);
	

	vector<Point2f> in_contour_f;
	vector<Point2f> out_contour_f;
	
	vector<Point> out_contour;
	vector<vector<Point>> out_contour_arr;
	
	//convert to float
	for(size_t i = 0; i < contour_.size(); i++)
		in_contour_f.push_back(contour_[i]);

	cv::perspectiveTransform(in_contour_f,out_contour_f, transform);	

	//convert back to int
	for(size_t i = 0; i < contour_.size(); i++)
		out_contour.push_back(out_contour_f[i]);
	return out_contour;
}

void ObjectType::drawScaled(Mat &image, RotatedRect roi) {
	vector<Point> draw_contour = transformContour(roi);
	vector<vector<Point>> draw_contour_arr;
	draw_contour_arr.push_back(draw_contour);
	drawContours(image,draw_contour_arr,0,Scalar(255), CV_FILLED);
}


void ObjectType::computeProperties()
{
	int min_x = numeric_limits<int>::max();
	int min_y = numeric_limits<int>::max();
	int max_x = numeric_limits<int>::min();
	int max_y = numeric_limits<int>::min();
	for (auto it = contour_.cbegin(); it != contour_.cend(); ++it)
	{
		min_x = std::min(min_x, it->x);
		min_y = std::min(min_y, it->y);
		max_x = std::max(max_x, it->x);
		max_y = std::max(max_y, it->y);
	}
	width_ = max_x - min_x;
	height_ = max_y - min_y;

	br_ = Rect(Point(min_x, min_y), Point(max_x, max_y));
	
	//to compute area the contour needs to be in floats
	vector<Point2f> contour_f;
	for(size_t i = 0; i < contour_.size(); i++)
		contour_f.push_back(contour_[i]);
	area_ = contourArea(contour_f);

	//compute moments and use them to find center of mass
	Moments mu = moments(contour_f, false);
	com_ = Point(mu.m10 / mu.m00, mu.m01 / mu.m00);

	com_ -= br_.tl();
}

bool ObjectType::operator== (const ObjectType &t1) const {
	return contour_ == t1.shape();
}

static Rect worldToScreenCoords(const Point3f &_position, ObjectType _type, const Point2f &fov_size, const Size &frame_size, float cameraElevation)
{
	float r = sqrtf(_position.x * _position.x + _position.y * _position.y + _position.z * _position.z) + (4.572 * 25.4)/1000.0;
	float azimuth = asinf(_position.x / sqrt(_position.x * _position.x + _position.y * _position.y));
	float inclination = asinf( _position.z / r ) + cameraElevation;

	Point2f percent_fov = Point2f(azimuth / fov_size.x, inclination / fov_size.y);
	Point2f dist_to_center(percent_fov.x * frame_size.width,
												 percent_fov.y * frame_size.height);

	Point2f rect_center(
			dist_to_center.x + (frame_size.width / 2.0),
			-dist_to_center.y + (frame_size.height / 2.0));

	Point2f angular_size( 2.0 * atan2f(_type.width() / 1000., (2.0*r)), 2.0 * atan2f(_type.height() / 1000.0, (2.0*r)));
	Point2f screen_size(
			angular_size.x * (frame_size.width / fov_size.x),
			angular_size.y * (frame_size.height / fov_size.y));

	Point topLeft(
			cvRound(rect_center.x - (screen_size.x / 2.0)),
			cvRound(rect_center.y - (screen_size.y / 2.0)));
			return Rect(topLeft.x, topLeft.y, cvRound(screen_size.x), cvRound(screen_size.y));
}


TrackedObject::TrackedObject(int id,
							 const ObjectType &type_in,
							 const Rect   &screen_position,
							 double            avg_depth,
							 Point2f       fov_size,
							 Size          frame_size,
							 float             camera_elevation,
							 float             dt,
							 float             accel_noise_mag,
							 size_t            historyLength) :
		_type(type_in),
		_detectHistory(historyLength),
		_positionHistory(historyLength),
		_KF(screenToWorldCoords(Point(screen_position.x+screen_position.width,screen_position.y+screen_position.height),
			avg_depth, fov_size, frame_size, camera_elevation),
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
	reverse(_id.begin(), _id.end());
}


TrackedObject::~TrackedObject()
{
}

// Set the position based on x,y,z coords

void TrackedObject::setPosition(const Point3f &new_position)
{
	_position = new_position;
	addToPositionHistory(_position);
}

// Set the position based on a rect on the screen and depth info from the zed
void TrackedObject::setPosition(const Rect &screen_position, double avg_depth,
		                        const Point2f &fov_size, const Size &frame_size)
{
	setPosition(screenToWorldCoords(Point(screen_position.x+screen_position.width,screen_position.y+screen_position.height), avg_depth, fov_size, frame_size, cameraElevation_));
}

#if 0
void TrackedObject::adjustPosition(const Eigen::Transform<double, 3, Eigen::Isometry> &delta_robot)
{
	//Eigen::AngleAxisd rot(0.5*M_PI, Eigen::Vector3d::UnitZ());

	Eigen::Vector3d old_pos_vec(_position.x, _position.y, _position.z);
	Eigen::Vector3d new_pos_vec = delta_robot * old_pos_vec;

	_position = Point3f(new_pos_vec[0], new_pos_vec[1], new_pos_vec[2]);

	for (auto it = _positionHistory.begin(); it != _positionHistory.end(); ++it)
	{
		Eigen::Vector3d old_pos_vector(it->x, it->y, it->z);
		Eigen::Vector3d new_pos_vector = delta_robot * old_pos_vector;
		*it = Point3f(new_pos_vector[0], new_pos_vector[1], new_pos_vector[2]);
	}
}
#endif
void TrackedObject::adjustPosition(const Mat &transform_mat, float depth, const Point2f &fov_size, const Size &frame_size)
{
	//get the position of the object on the screen
	Rect screen_rect = getScreenPosition(fov_size,frame_size);
	Point screen_pos = Point(screen_rect.tl().x + screen_rect.width / 2, screen_rect.tl().y + screen_rect.height / 2);

	//create a matrix to hold positon for matrix multiplication
	Mat pos_mat(3,1,CV_64FC1);
	pos_mat.at<double>(0,0) = screen_pos.x;
	pos_mat.at<double>(0,1) = screen_pos.y;
	pos_mat.at<double>(0,2) = 1.0;

	//correct the position
	Mat new_screen_pos_mat(3,1,CV_64FC1);
	new_screen_pos_mat = transform_mat * pos_mat;
	Point new_screen_pos = Point(new_screen_pos_mat.at<double>(0),new_screen_pos_mat.at<double>(1));

	//create a dummy bounding rect because setPosition requires a bounding rect as an input rather than a point
	Rect new_screen_rect(new_screen_pos.x,new_screen_pos.y,0,0);
	setPosition(new_screen_rect,depth,fov_size,frame_size);
	//update the history
	for (auto it = _positionHistory.begin(); it != _positionHistory.end(); ++it)
	{
		screen_rect = worldToScreenCoords(*it,_type,fov_size,frame_size, cameraElevation_);
		screen_pos = Point(screen_rect.tl().x + screen_rect.width / 2, screen_rect.tl().y + screen_rect.height / 2);
		pos_mat.at<double>(0,0) = screen_pos.x;
		pos_mat.at<double>(0,1) = screen_pos.y;
		pos_mat.at<double>(0,2) = 1.0;
		Mat new_screen_pos_mat = transform_mat * pos_mat;
		Point new_screen_pos = Point(new_screen_pos_mat.at<double>(0),new_screen_pos_mat.at<double>(1));
		*it = screenToWorldCoords(new_screen_pos, depth, fov_size, frame_size, cameraElevation_);
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
void TrackedObject::addToPositionHistory(const Point3f &pt)
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
		detectRatio = min(0.01, detectRatio);
	return detectRatio;
}


Rect TrackedObject::getScreenPosition(const Point2f &fov_size, const Size &frame_size) const
{
	return worldToScreenCoords(_position, _type, fov_size, frame_size, cameraElevation_);
}



Point3f TrackedObject::predictKF(void)
{
	return _KF.GetPrediction();
}


Point3f TrackedObject::updateKF(Point3f pt)
{
	return _KF.Update(pt);
}

#if 0
void TrackedObject::adjustKF(const Eigen::Transform<double, 3, Eigen::Isometry> &delta_robot)
{
	_KF.adjustPrediction(delta_robot);
}
#endif
void TrackedObject::adjustKF(Point3f delta_pos)
{
	_KF.adjustPrediction(delta_pos);
}


//Create a tracked object list
// those stay constant for the entire length of the run
TrackedObjectList::TrackedObjectList(const Size &imageSize, const Point2f &fovSize, float cameraElevation) :
	_detectCount(0),
	_imageSize(imageSize),
	_fovSize(fovSize),
	_cameraElevation(cameraElevation)
{
}
#if 0
// Adjust position for camera motion between frames using fovis
void TrackedObjectList::adjustLocation(const Eigen::Transform<double, 3, Eigen::Isometry> &delta_robot)
{
	for (auto it = _list.begin(); it != _list.end(); ++it)
	{
		it->adjustPosition(delta_robot);
		it->adjustKF(delta_robot);
	}
}
#endif
// Adjust position for camera motion between frames using optical flow
void TrackedObjectList::adjustLocation(const Mat &transform_mat)
{
	for (auto it = _list.begin(); it != _list.end(); ++it)
	{
		//measure the amount that the position changed and apply the same change to the kalman filter
		Point3f old_pos = it->getPosition();
		//compute r and use it for depth (assume depth doesn't change)
		float r = sqrt(it->getPosition().x * it->getPosition().x + it->getPosition().y * it->getPosition().y + it->getPosition().z * it->getPosition().z);
		it->adjustPosition(transform_mat,r,_fovSize,_imageSize);
		Point3f delta_pos = it->getPosition() - old_pos;

		it->adjustKF(delta_pos);
	}
}

// Simple printout of list into stdout
void TrackedObjectList::print(void) const
{
	for (auto it = _list.cbegin(); it != _list.cend(); ++it)
	{
		cout << it->getId() << " location ";
		Point3f position = it->getPosition();
		cout << "(" << position.x << "," << position.y << "," << position.z << ")" << endl;
	}
}

// Return list of detect info for external processing
void TrackedObjectList::getDisplay(vector<TrackedObjectDisplay> &displayList) const
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
void TrackedObjectList::processDetect(const vector<Rect> &detectedRects,
									  const vector<float> depths,
									  const vector<ObjectType> &types)
{
	if (detectedRects.size() || _list.size())
		cout << "---------- Start of process detect --------------" << endl;
	print();
	vector<Point3f> detectedPositions;
	if (detectedRects.size() > 0)
		cout << detectedRects.size() << " objects" << endl;
	for (size_t i = 0; i < detectedRects.size(); i++)
	{
		Point rect_center = Point(detectedRects[i].x+detectedRects[i].width,detectedRects[i].y+detectedRects[i].height);
		detectedPositions.push_back(
				screenToWorldCoords(rect_center, depths[i], _fovSize, _imageSize, _cameraElevation));
		cout << "Detected rect [" << i << "] = " << detectedRects[i] << " positions[" << detectedPositions.size() - 1 << "]:" << detectedPositions[detectedPositions.size()-1] << endl;
	}
	// TODO :: Combine overlapping detections into one?

	// Maps tracks to the closest new detected object.
	// assignment[track] = index of closest detection
	vector<int> assignment;
	if (_list.size())
	{
		size_t tracks = _list.size();		          // Number of tracks
		size_t detections = detectedPositions.size(); //  number of detections

		vector< vector<double> > Cost(tracks,vector<double>(detections));

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
					Point3f diff = it->getPosition() - detectedPositions[d];
					Cost[t][d] = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
				} else {
					Cost[t][d] = numeric_limits<float>::max();
				}
			}
		}

		// Solving assignment problem (find minimum-cost assignment
		// between tracks and previously-predicted positions)
		AssignmentProblemSolver APS;
		APS.Solve(Cost, assignment, AssignmentProblemSolver::optimal);

		cout << "After APS : "<<endl;
		for(size_t i = 0; i < assignment.size(); i++)
			cout << i << ":" << assignment[i] << endl;
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
			cout << "New assignment created " << i << endl;
			_list.push_back(TrackedObject(_detectCount++,types[i], detectedRects[i], depths[i], _fovSize, _imageSize));
		}
	}

	auto tr = _list.begin();
	auto as = assignment.begin();
	while ((tr != _list.end()) && (as != assignment.end()))
	{
		// If track updated less than one time, than filter state is not correct.
		cout << "Predict: " << endl;
		Point3f prediction = tr->predictKF();
		cout << "prediction:" << prediction << endl;

		if(*as != -1) // If we have assigned detect, then update using its coordinates
		{
			cout << "Update match: " << endl;
			tr->setPosition(tr->updateKF(detectedPositions[*as]));
			cout << tr->getScreenPosition(_fovSize, _imageSize) << endl;
			tr->setDetected();
		}
		else          // if not continue using predictions
		{
			cout << "Update no match: " << endl;
			tr->setPosition(tr->updateKF(prediction));
			cout << tr->getScreenPosition(_fovSize, _imageSize) << endl;
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
			//cout << "Dropping " << it->getId() << endl;
			it = _list.erase(it);
		}
		else
		{
			++it;
		}
	}
	print();
	if (detectedRects.size() || _list.size())
		cout << "---------- End of process detect --------------" << endl;
}
