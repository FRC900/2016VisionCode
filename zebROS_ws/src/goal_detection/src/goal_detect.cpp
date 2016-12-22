#include <iostream>
#include <opencv2/opencv.hpp>
#include <zmq.hpp>

#include "zedcamerain.hpp"
#include "zedsvoin.hpp"
#include "zmsin.hpp"
#include "GoalDetector.hpp"
#include "Utilities.hpp"
#include "track3d.hpp"
#include "frameticker.hpp"

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float64MultiArray.h"

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <sstream>


using namespace cv;
using namespace std;

namespace cv_bridge {

class CvImage
{
public:
  std_msgs::Header header;
  std::string encoding;
  cv::Mat image;
};

typedef boost::shared_ptr<CvImage> CvImagePtr;
typedef boost::shared_ptr<CvImage const> CvImageConstPtr;

}

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
	Mat image;
	Mat depth;
	//Mat depthNorm;
	Rect bound;
	FrameTicker frameTicker;
GoalDetector* gd(Point2f(cap->getCameraParams().fov.x, 
				            cap->getCameraParams().fov.y), 
			        Size(cap->width(),cap->height()), true);
  
public:
  ImageConverter()
    : it_(nh_)
  {
	gd = new GoalDetector(Point2f(cap->getCameraParams().fov.x, 
				            cap->getCameraParams().fov.y), 
			        Size(cap->width(),cap->height()), true);

    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/camera/image_raw", 1, 
      &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);

    cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // Draw an example circle on the video stream
    if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
      cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::waitKey(3);
    
    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
  }
};

int main(int argc, char **argv)
{
	MediaIn *cap = NULL;
	if (argc == 2)
		cap = new ZMSIn(argv[1]);
	else
		cap = new ZedCameraIn(false);

	if (cap == NULL)
	{
		cerr << "Error creating input" << endl;
		return -1;
	}

	image_transport::ImageTransport it(nh);


	cv_bridge::CvImagePtr cv_ptr;
	try
	{
		cv_ptr = cv_bridge::toCvCopy()
	}

	

	// zmq::context_t context(1);
	// zmq::socket_t publisher(context, ZMQ_PUB);
	
	ros::init(argc, argv, "goal_tracker");
	ros::NodeHandle n;

	ros::Publisher goal_pub = n.advertise<std_msgs::String>("goal_data", 1000);
	ros::Rate loop_rate(10);

	// std::cout<< "Starting network publisher 5800" << std::endl;
	// publisher.bind("tcp://*:5800");

	
	while (ros::ok())
	{
		frameTicker.mark();
		//imshow ("Normalized Depth", depthNorm);

		gd.processFrame(image, depth);
		gd.drawOnFrame(image);

		stringstream ss;
		ss << fixed << setprecision(2) << frameTicker.getFPS() << "FPS";
		putText(image, ss.str(), Point(image.cols - 15 * ss.str().length(), 50), FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,255));
		rectangle(image, gd.goal_rect(), Scalar(255,0,0), 2);
		imshow ("Image", image);

		std_msgs::String msg;

		stringstream gString;
		gString << "G ";
		gString << fixed << setprecision(4) << gd.dist_to_goal() << " ";
		gString << fixed << setprecision(2) << gd.angle_to_goal();

		cout << "G : " << gString.str().length() << " : " << gString.str() << endl;
		
		msg.data = gString.str();
				
		
		// zmq::message_t grequest(gString.str().length() - 1);
		// memcpy((void *)grequest.data(), gString.str().c_str(), gString.str().length() - 1);
		// publisher.send(grequest);
		goal_pub.publish(msg);
		ros::spinOnce();

		loop_rate.sleep();
		

		if ((uchar)waitKey(5) == 27)
		{
			break;
		}
	}
	return 0;
}
