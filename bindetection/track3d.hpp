#ifndef TRACK_HPP_INC__
#define TRACK_HPP_INC__

#include <algorithm>
#include <string>
#include <list>
#include <Eigen/Geometry>

const size_t TrackedObjectHistoryLength = 20;
const size_t TrackedObjectDataLength = 5;

const double HFOV = 69; // horizontal field of view of C920 camera


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
  ObjectType(std::vector< cv::Point2f > contour_in);

  //get the contour associated with the object type. Useful for shape comparison
  std::vector< cv::Point2f > shape () const { return _contour; }

  //get physical characteristics
  cv::Point2f com () const { return _com; }
  float width () const {return _width; }
  float height () const {return _height; }
  float area () const { return _area; }

private:

  std::vector< cv::Point2f > _contour;

  //properties are computed and stored internally so that they don't have to be recomputed
  //every time the get functions are called
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
    //rect area is the area of the bounding rectangle
    double rectArea(void) const;

    //contour area is the area of the contour stored in ObjectType
    //scaled into the bounding rect
    //only different in cases where contour is not a rectangle
    double contourArea(void) const; //P.S. underestimates slightly

    // Update current object position based on a 3d position or
    //input rect on screen and depth
    void setPosition(const cv::Point3f &new_position);
    void setPosition(const cv::Rect &screen_position, const double avg_depth);

    //get position of a rect on the screen corresponding to the object size and location
    //inverse of setPosition(Rect,depth)
    cv::Rect getScreenPosition() const;
    cv::Point3f getPosition() const;

    //averages the position over the past frames
    //variance is given separately for X,Y,Z
    cv::Point3f getAveragePosition(cv::Point3f &variance) const;

    //aggregates the variance data into a single value
    //not sure if this or the top function is more useful
    cv::Point3f getAveragePosition(double &variance) const;

    //how many frames ago the object was last seen
    int lastSeen(void);

    std::string getId(void) const { return _id; }

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

    //runtime constants needed for computing position from rect
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
  // the rotation of the robot straight from fovis
  void adjustLocation(const Eigen::Transform<double, 3, Eigen::Isometry> &delta_robot);

  // Simple printout of list into
  void print(void) const;

  // Return list of detect info for external processing
  void getDisplay(std::vector<TrackedObjectDisplay> &displayList) const;

  // Process a detected rectangle from the current frame.
  // This will either match a previously detected object or
  // if not, add a new object to the list
  void processDetect(const cv::Rect &detectedRect, ObjectType type);

  private :
  std::list<TrackedObject> _list;        // list of currently valid detected objects
  int _detectCount; // ID of next objectcreated
  double _objectWidth; // width of the object tracked

  //values stay constant throughout the run but are needed for computing stuff
  cv::Size _imageSize;
  cv::Size _fovSize;
};

#endif
