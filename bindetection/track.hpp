#ifndef TRACK_HPP_INC__
#define TRACK_HPP_INC__

#include <algorithm>
#include <string>
#include <list>

const size_t TrackedObjectHistoryLength = 20;
const size_t TrackedObjectDataLength = 5;

const double HFOV = 69; // horizontal field of view of C920 camera

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
      TrackedObject(const cv::Rect &position, 
	    int id,
	    size_t historyLength = TrackedObjectHistoryLength,
	    size_t dataLength = TrackedObjectDataLength);

      // Copy constructor and assignement operators are needed to do a
      // deep copy.  This makes new arrays for each object copied rather
      TrackedObject(const TrackedObject &object);
      TrackedObject &operator=(const TrackedObject &object);
      ~TrackedObject();

	  // Adjust the expected object position to account
	  // for camera motion between last frame and current one
      void adjustPosition(const cv::Mat &transformMat);

      // Set the distance to the bin for the current frame
      void setDistance(double distance);
      
      // Set the distance to the target using the detected
      // rectangle plus the known size of the object and frame size
      void setDistance(const cv::Rect &rect, double objWidth, int imageWidth);

      // Set the angle off center for the current frame
      void setAngle(double angle);

      // Set the angle off center for the current frame
      // using the detected rectangle.
      void setAngle(const cv::Rect &rect, int imageWidth);

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

      // Return the distance in pixels between the 
      // tracked object's position and a point
      double distanceFromPoint(cv::Point point) const;

      // Return the area of the tracked object
      double area(void) const;
      
      // Return the position of the tracked object
      cv::Rect getPosition(void) const;

      // Update current object position
      // Maybe maintain a range of previous positions seen +/- some margin instead?
      cv::Rect setPosition(const cv::Rect &position);

      double getAverageDistance(double &stdev) const;
      double getAverageAngle(double &stdev) const;

      std::string getId(void) const;
      
   private :
      cv::Rect _position;   // last position of tracked object
      size_t   _listLength; // number of entries in history arrays
      size_t   _dataLength; // number of entries in history arrays
      size_t   _listIndex;  // current entry being modified in history arrays
      // whether or not the object was seen in a given frame - 
      // used to flag entries in other history arrays as valid 
      // and to figure out which tracked objects are persistent 
      // enough to care about
      bool    *_detectArray;  

      // Arrays of data for distance and angle
      double  *_distanceArray;
      double  *_angleArray;
      std::string _id; //unique target ID - use a string rather than numbers so it isn't confused
                       // with individual frame detect indexes

      // Helper function to average distance and angle
      double getAverageAndStdev(double *list, double &stdev) const;
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
      TrackedObjectList(double objectWidth, int imageWidth);
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
      void adjustPosition(const cv::Mat &transformMat);

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
      int                      _imageWidth;  // width of captured frame
      int                      _detectCount; // ID of next objectcreated
      double                   _objectWidth; // width of the object tracked
};

#endif

