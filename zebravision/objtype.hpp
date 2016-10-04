#pragma once 

#include <vector>
#include "opencv2_3_shim.hpp"

//class to hold the type of object that a detected object is
//main information is the contour that is the shape of the object
//stores properties to make matching easy
class ObjectType 
{
	public:
		//in this constructor there are contours prebuilt for game objects
		//1 - ball (2016 Game)
		//2 - bin (2015 Game)
		//// TODO : turn this into an enum 
		ObjectType(int contour_type_id);

		//this constructor takes a custom contour
		ObjectType(const std::vector< cv::Point2f > &contour_in);
		ObjectType(const std::vector< cv::Point > &contour_in);

		//get the contour associated with the object type. 
		// Useful for shape comparison
		std::vector< cv::Point2f > shape (void) const { return contour_; }

		//get physical characteristics
		cv::Point2f com (void) const { return com_; }
		float width (void) const {return width_; }
		float height (void) const {return height_; }
		float area (void) const { return area_; }
		float boundingArea (void) const { return width_ * height_; }

		//comparison operator overload just checks if the contours are equal
		bool operator== (const ObjectType &t1) const;

	private:
		std::vector< cv::Point2f > contour_;

		// properties are computed and stored internally so that they
		// don't have to be recomputed every time the get functions are called
		float width_;
		float height_;
		float area_;
		cv::Point2f com_; //center of mass

		//called by constructor to compute properties
		void computeProperties(void);
};

