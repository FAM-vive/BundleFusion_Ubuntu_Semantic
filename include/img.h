#include <iostream>
#include <string>
#include <ros/ros.h>
#include <ros/master.h>
#include <ros/package.h>
#include <ros/macros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <image_transport/image_transport.h>
#include <color_depth_msg/colordepth.h>
#include <opencv2/opencv.hpp>

void RosInit(int argc,char** argv);

class ImageSaver{
	cv::Mat color;
	cv::Mat depth;
	ros::NodeHandle n;
	ros::Subscriber depthSubscriber;
public:
	ImageSaver();
	~ImageSaver();
	void callbackDepth(const color_depth_msg::colordepthConstPtr& msg);
	cv::Mat getColor();
	cv::Mat getDepth();
	void imageSpin();

};
