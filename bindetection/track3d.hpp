#ifndef TRACK_HPP_INC__
#define TRACK_HPP_INC__

#include <algorithm>
#include <string>
#include <list>
#include <Eigen/Geometry>
#include "kalman.hpp"

const size_t TrackedObjectHistoryLength = 20;

//class to hold the type of object that a detected object is
//main information is the contour that is the shape of the object
//stores properties to make matching easy
class ObjectType {

	public:
		//in this constructor there are contours prebuilt for game objects
		//1 - ball (2016 Game)
		//2 - bin (2015 Game)
		ObjectType(int contour_type_id);

		//this constructor takes a custom contour
		ObjectType(const std::vector< cv::Point2f > &contour_in);
		ObjectType(const std::vector< cv::Point > &contour_in);

		//get the contour associated with the object type. Useful for shape comparison
		std::vector< cv::Point2f > shape (void) const { return _contour; }

		//get physical characteristics
		cv::Point2f com (void) const { return _com; }
		float width (void) const {return _width; }
		float height (void) const {return _height; }
		float area (void) const { return _area; }
		float boundingArea (void) const { return _width * _height; }

		//comparison operator overload just checks if the contours are equal
		bool operator== (const ObjectType &t1) const;

	private:
		std::vector< cv::Point2f > _contour;

		// properties are computed and stored internally so that they
		// don't have to be recomputed every time the get functions are called
		float _width;
		float _height;
		float _area;
		cv::Point2f _com; //center of mass

		//called by constructor to compute properties
		void computeProperties(void);
};


// Class to hold info on a tracked object
// Keeps a history per previous <historyLength> frames
// of whether the object was seen or not, and the
// calcualated distance and angle for each
// detection.
// Keep track of position (in 3 dimensions) of last detection - used
// to compare against new hits to see if this is the
// same object.
// Has method to compensate for robot rotation and translation with
// data from the fovis code
class TrackedObject
{
	public :
		TrackedObject( int id,
				const ObjectType &type_in,
				const cv::Rect   &screen_position,
				double            avg_depth,
				cv::Point2f       fov_size,
				cv::Size          frame_size,
				float             dt = 0.5,
				float             accel_noise_mag = 0.25,
				size_t            historyLength = TrackedObjectHistoryLength);

		~TrackedObject();

		// Mark the object as detected in this frame
		void setDetected(void);

		// Clear the object detect flag for this frame.
		// Probably should only happen when moving to a new
		// frame, but may be useful in other cases
		void clearDetected(void);

		// Return the percent of last _listLength frames
		// the object was seen
		double getDetectedRatio(void) const;

		bool tooManyMissedFrames(void) const;

		// Increment to the next frame
		void nextFrame(void);

		//contour area is the area of the contour stored in ObjectType
		//scaled into the bounding rect
		//only different in cases where contour is not a rectangle
		double contourArea(const cv::Point2f &fov_size, const cv::Size &frame_size) const; //P.S. underestimates slightly

		// Update current object position based on a 3d position or
		//input rect on screen and depth
		void setPosition(const cv::Point3f &new_position);
		void setPosition(const cv::Rect &screen_position, double avg_depth, const cv::Point2f &fov_size, const cv::Size &frame_size);

		// Adjust tracked object position based on motion
		// of the camera
		void adjustPosition(const Eigen::Transform<double, 3, Eigen::Isometry> &delta_robot);


		//get position of a rect on the screen corresponding to the object size and location
		//inverse of setPosition(Rect,depth)
		cv::Rect getScreenPosition(const cv::Point2f &fov_size, const cv::Size &frame_size) const;
		cv::Point3f getPosition(void) const { return _position; }

		void adjustKF(const Eigen::Transform<double, 3, Eigen::Isometry> &delta_robot);
		cv::Point3f predictKF(void);
		cv::Point3f updateKF(cv::Point3f pt);

		std::string getId(void) const { return _id; }
		ObjectType getType(void) const { return _type; }

	private :
		ObjectType _type;

		cv::Point3f _position; // last position of tracked object
		size_t   _historyIndex;   // current entry being modified in history arrays
		// whether or not the object was seen in a given frame -
		// used to flag entries in other history arrays as valid
		// and to figure out which tracked objects are persistent
		// enough to care about
		std::vector<bool>        _detectHistory;
		std::vector<cv::Point3f> _positionHistory;

		// Kalman filter for tracking and noise filtering
		TKalmanFilter _KF;

		std::string _id; //unique target ID - use a string rather than numbers so it isn't confused
						 // with individual frame detect indexes
						 //
		int missedFrameCount_;
		size_t positionHistoryMax_;

		void addToPositionHistory(const cv::Point3f &pt);
};

  // Used to return info to display
  struct TrackedObjectDisplay
  {
    std::string id;
    cv::Rect rect;
    double ratio;
    cv::Point3f position;
  };

  // Tracked object array -
  //
  // Need to create array of tracked objects.
  // For each frame,
  //   use fovis data to determine camera translation and rotation
  //   update each object's position to "undo" that motion
  //   for each detected rectangle
  //      try to find a close match in the list of previously detected objects
  //      if found
  //         update that entry's distance and angle
  //      else
  //         add new entry
  //   find a way to clear out images "lost" - look at history, how far
  //   off the screen is has been rotated, etc.  Don't be too aggressive
  //   since we could rotate back and "refind" an object which has disappeared
  //
class TrackedObjectList
{
	public :
		// Create a tracked object list.  Set the object width in inches
		// (feet, meters, parsecs, whatever) and imageWidth in pixels since
		// those stay constant for the entire length of the run
		TrackedObjectList(const cv::Size &imageSize, const cv::Point2f &fovSize);

		// Adjust the angle of each tracked object based on
		// the rotation of the robot straight from fovis
		void adjustLocation(const Eigen::Transform<double, 3, Eigen::Isometry> &delta_robot);

		// Simple printout of list into
		void print(void) const;

		// Return list of detect info for external processing
		void getDisplay(std::vector<TrackedObjectDisplay> &displayList) const;

		// Process a set of detected rectangles
		// Each will either match a previously detected object or
		// if not, be added as new object to the list
		void processDetect(const std::vector<cv::Rect> &detectedRects,
						   const std::vector<float> depths,
						   const std::vector<ObjectType> &types,
						   const Eigen::Transform<double, 3, Eigen::Isometry> &delta_robot
							);

	private :
		std::list<TrackedObject> _list; // list of currently valid detected objects
		int _detectCount;               // ID of next object to be created
		double _objectWidth;            // width of the object tracked

		//values stay constant throughout the run but are needed for computing stuff
		cv::Size    _imageSize;
		cv::Point2f _fovSize;
};

#endif
