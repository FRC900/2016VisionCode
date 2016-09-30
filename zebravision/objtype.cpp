// A class for defining objects we're trying to 
// detect.  The class stores information about shape 
// and size of the objects in real-world measurements
#include "objtype.hpp"

using namespace std;
using namespace cv;

ObjectType::ObjectType(int contour_type_id=1) {
	switch(contour_type_id) {
		//loads one of the preset shapes into the

		case 1: //a ball!
			{
				float ball_diameter = 0.2476; // meters
				contour_.push_back(Point2f(0,0));
				contour_.push_back(Point2f(0,ball_diameter));
				contour_.push_back(Point2f(ball_diameter,ball_diameter));
				contour_.push_back(Point2f(ball_diameter,0));
			}
			break;

		case 2: //a bin (just because)
			contour_.push_back(Point2f(0,0));
			contour_.push_back(Point2f(0,0.5842));
			contour_.push_back(Point2f(0.5842,0.5842));
			contour_.push_back(Point2f(0.5842,0));
			break;

		case 3: //the vision goal
			{
				float max_y = .3048;
				contour_.push_back(Point2f(0, max_y - 0));
				contour_.push_back(Point2f(0, max_y - 0.3048));
				contour_.push_back(Point2f(0.0508, max_y - 0.3048));
				contour_.push_back(Point2f(0.0508, max_y - 0.0508));
				contour_.push_back(Point2f(0.508-0.0508, max_y - 0.0508));
				contour_.push_back(Point2f(0.508-0.0508, max_y - 0.3048));
				contour_.push_back(Point2f(0.508, max_y - 0.3048));
				contour_.push_back(Point2f(0.508, max_y - 0));
			}
			break;

		default:
			cerr << "error initializing object!" << endl;
	}

	computeProperties();

}

ObjectType::ObjectType(const vector< Point2f > &contour_in) :
	contour_(contour_in)
{
	computeProperties();
}

ObjectType::ObjectType(const vector< Point > &contour_in)
{
	for(size_t i = 0; i < contour_in.size(); i++)
	{
		Point2f p;
		p.x = (float)contour_in[i].x;
		p.y = (float)contour_in[i].y;
		contour_.push_back(p);
	}
	computeProperties();

}

void ObjectType::computeProperties()
{
	float min_x = numeric_limits<float>::max();
	float min_y = numeric_limits<float>::max();
	float max_x = numeric_limits<float>::min();
	float max_y = numeric_limits<float>::min();
	for (auto it = contour_.cbegin(); it != contour_.cend(); ++it)
	{
		min_x = min(min_x, it->x);
		min_y = min(min_y, it->y);
		max_x = max(max_x, it->x);
		max_y = max(max_y, it->y);
	}
	width_ = max_x - min_x;
	height_ = max_y - min_y;
	area_ = contourArea(contour_);

	//compute moments and use them to find center of mass
	Moments mu = moments(contour_, false);
	com_ = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
}

bool ObjectType::operator== (const ObjectType &t1) const 
{
	return this->shape() == t1.shape();
}


