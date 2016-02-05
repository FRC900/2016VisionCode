#include <algorithm>
#include <string>
#include <list>

const size_t TrackedObjectHistoryLength = 20;
const size_t TrackedObjectDataLength = 5;


//class to hold the type of object that a detected object is
//main information is the contour that is the shape of the object
class ObjectType {

public:
  //delta_rotate is whether the object skews when looking at it not straight on. Bins and balls don't but vision goals do.
  ObjectType(int contour_type_id); //allows initialization using a prebuilt contour
  ObjectType(vector< cv::Point2f > contour_in, bool delta_rotate_in=false);

  vector< cv::Point2f > shape () const { return _contour; }
  cv::Point2f com () const { return _width; }
  float width () const {return _width; }
  float height () const {return _height; }
  float area () const { return _area; }

private:
  void computeProperties(void); //called by constructor
  vector< cv::Point2f > _contour;
  cv::Point2f _com; //center of mass
  float _width;
  float _height; //precomputed in header and stored so that they don't have to be recomputed every
  float _area; //time you want one of the properties
  bool _delta_rotate;

};

class TrackedObject
{
   public :
      // IDs are assigned by the constructor into the variable given
      //constructor can take either a screen position or 2d position relative to robot
      TrackedObject(ObjectType type_in, const cv::Rect &screen_position, int &id);
      TrackedObject(ObjectType type_in, const cv::Point2f &position_in, int &id);

      // Copy constructor and assignement operators are needed to do a
      // deep copy.  This makes new arrays for each object copied rather
      TrackedObject(const TrackedObject &object);
      TrackedObject &operator=(const TrackedObject &object);
      ~TrackedObject();

      void setPosition(cv::Point2f new_location);

      //called by Tracker
      //both of these are how much the ROBOT has changed not how much the OBJECT has changed (negative of each other)
      void changePosition(cv::Point2f delta_position); //takes the amount the robot has translated and corrects the bin to the new location
      void changeRotation(double delta_rotation); //takes the amount the robot has rotated and corrects the bin to the new location


      // Set the distance to the target using the detected
      // rectangle plus the known size of the object and frame size
      void setDistance(const cv::Rect &rect, double objWidth, int imageWidth);

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

      // Return the position of the tracked object
      cv::Rect getScreenPosition(void) const;
      cv::Point2f getPosition(void) const;

      double getAverageDistance(double &stdev) const;
      double getAverageAngle(double &stdev) const;

      std::string getId(void) const;

   private :
      cv::Point2f _position;   // last position of tracked object
      size_t   _listLength; // number of entries in history arrays
      size_t   _dataLength; // number of entries in history arrays
      size_t   _listIndex;  // current entry being modified in history arrays
      // whether or not the object was seen in a given frame -
      // used to flag entries in other history arrays as valid
      // and to figure out which tracked objects are persistent
      // enough to care about
      bool    *_detectArray;

      std::string _id; //unique target ID - use a string rather than numbers so it isn't confused
                       // with individual frame detect indexes
};

// Tracked object array -
//
// Need to create array of tracked objects.
// For each frame,
//   read the angle the robot has turned (adjustAngle)
//   update each object's position with that angle :
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
class Tracker
{
   public :
      // Create a tracked object list.  Set the object width in inches
      // (feet, meters, parsecs, whatever) and imageWidth in pixels since
      // those stay constant for the entire length of the run
      Tracker();
      // Go to the next frame.  First remove stale objects from the list
      // and call nextFrame on the remaining ones
      void nextFrame(void);

      // Adjust the angle of each tracked object based on
      // the rotation of the robot
      void adjustAngle(double deltaAngle);
      void adjustPosition(cv::Point2f deltaPosition)

      // Simple printout of list into
      void print(void) const;

      // Return list of detect info for external processing
      void getAll(std::vector<TrackedObject> &displayList) const;
      void getObject(int id, TrackedObject &objectOut) const;

      // Process a detected rectangle from the current frame.
      // This will either match a previously detected object or
      // if not, add a new object to the list
      void processDetect(const cv::Rect &detectedRect, const int object_type_id); //most commonly used
      void processDetect(const cv::Rect &detectedRect, vector <cv::Point2f> object_contour, bool delta_rotate_in);

   private :
      std::list<TrackedObject> _objects;        // list of currently valid detected objects
};

#endif
