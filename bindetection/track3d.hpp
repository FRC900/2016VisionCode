#ifndef TRACK_HPP_INC__
#define TRACK_HPP_INC__

#include <algorithm>
#include <string>
#include <list>

const size_t TrackedObjectHistoryLength = 20;
const size_t TrackedObjectDataLength = 5;

const double HFOV = 69; // horizontal field of view of C920 camera


//class to hold the type of object that a detected object is
//main information is the contour that is the shape of the object
class ObjectType {

public:
  //delta_rotate is whether the object skews when looking at it not straight on. Bins and balls don't but vision goals do.
  ObjectType(int contour_type_id); //allows initialization using a prebuilt contour
  ObjectType(vector< cv::Point3f > contour_in, bool delta_rotate_in=false);

  vector< cv::Point3f > shape () const { return _contour; }
  cv::Point3f com () const { return _width; }
  float width () const {return _width; }
  float height () const {return _height; }
  float area () const { return _area; }

private:
  void computeProperties(void); //called by constructor
  vector< cv::Point3f > _contour;
  cv::Point3f _com; //center of mass
  float _width;
  float _height; //precomputed in header and stored so that they don't have to be recomputed every
  float _area; //time you want one of the properties

};





// Class to hold info on a tracked object
// Keeps a history per previous <historyLength> frames
// of whether the object was seen or not, and the 
// calcualated distance and angle for each 
// detection.
// Keep track of position of last detection - used
// to compare against new hits to see if this is the
// same object.
// Has method to rotate the position in the x direction
// to account for robot movement 
class TrackedObject
{
   public :
      TrackedObject( int id,
		ObjectType &type_in,
		cv::Size2f fov_size,
		cv::Size2f frame_size,
	    size_t historyLength = TrackedObjectHistoryLength,
	    size_t dataLength = TrackedObjectDataLength);

      // Copy constructor and assignement operators are needed to do a
      // deep copy.  This makes new arrays for each object copied rather
      TrackedObject(const TrackedObject &object);
      TrackedObject &operator=(const TrackedObject &object);
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

      // Increment to the next frame
      void nextFrame(void);

      // Return the area of the tracked object
      double rectArea(void) const;
	  double contourArea(void) const;

      // Update current object position
      // Maybe maintain a range of previous positions seen +/- some margin instead?
      void setPosition(const cv::Point3f &new_position);
	  void setPosition(const cv::Rect &screen_position, const double avg_depth);

	  cv::Rect getScreenPosition() const;
	  cv::Point3f getPosition() const;

      cv::Point3f getAveragePosition(cv::Point3f &variance) const;
	  cv::Point3f getAveragePosition(double &variance) const; //aggregate variance

	  int lastSeen(void);

      std::string getId(void) const;
      
   private :

	  ObjectType _type;	
	
      cv::Point3f _position;   // last position of tracked object
      size_t   _listLength; // number of entries in history arrays
      size_t   _dataLength; // number of entries in history arrays
      size_t   _listIndex;  // current entry being modified in history arrays
      // whether or not the object was seen in a given frame - 
      // used to flag entries in other history arrays as valid 
      // and to figure out which tracked objects are persistent 
      // enough to care about
      bool    *_detectArray;  

	  cv::Size2f _fov_size;
	  cv::Size2f _frame_size;

      // Arrays of data for position
      cv::Point3f  *_positionArray;
      std::string _id; //unique target ID - use a string rather than numbers so it isn't confused
                       // with individual frame detect indexes
};

// Used to return info to display
struct TrackedObjectDisplay
{
   std::string id;
   cv::Rect rect;
   double ratio;
   double distance;
   double angle;
};

// Tracked object array - 
// 
// Need to create array of tracked objects.
// For each frame, 
//   use optical flow to figure out camera motions
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
      TrackedObjectList(cv::Size imageSize, cv::Size fovSize);
#if 0
      void Add(const cv::Rect &position)
      {
	 _list.push_back(TrackedObject(position));
      }
#endif
      // Go to the next frame.  First remove stale objects from the list
      // and call nextFrame on the remaining ones
      void nextFrame(void);
      
      // Adjust the angle of each tracked object based on
      // the rotation of the robot
      void adjustLocation(const &Eigen::Isometry3d &delta_robot);

      // Simple printout of list into
      void print(void) const;

      // Return list of detect info for external processing
      void getDisplay(std::vector<TrackedObjectDisplay> &displayList) const;

      // Process a detected rectangle from the current frame.
      // This will either match a previously detected object or
      // if not, add a new object to the list
      void processDetect(const cv::Rect &detectedRect);

   private :
      std::list<TrackedObject> _list;        // list of currently valid detected objects
      cv::Size                      _imageSize;  // width of captured frame
	  cv::Size 					_fovSize;
      int                      _detectCount; // ID of next objectcreated
      double                   _objectWidth; // width of the object tracked
};

#endif

