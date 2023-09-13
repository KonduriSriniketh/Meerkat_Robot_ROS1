#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include "std_msgs/Int32.h"
#include "nav_msgs/Odometry.h"
#include "meerkat_msgs/Speed.h"
#include "geometry_msgs/Quaternion.h"
#include "sensor_msgs/Joy.h"
#include "sensor_msgs/Imu.h"
#include <tf/transform_broadcaster.h>
#include <iostream>
#include <cmath>


double x = 0.0;
double y = 0.0;
double th = 0.0;
double vx = 0.0;
double vy = 0.0;
double vth = 0.0;
double _wheel_base = 0.39;
double _wheel_radius = 0.0625;
double _left_speed;
double _right_speed;
double cb_timeout;

double theta;
double angular_velocity;

int _l_rate;
ros::Time cb_timer;
bool broadcast_odom_tf;
std::string _odom_topic_name;
std::string _speed_pub_name;
std::string _paw_frame_name;
std::string _imu_topic_name;

geometry_msgs::Quaternion _orientation_data;


void speedCb(const meerkat_msgs::Speed::ConstPtr& speed)
{
  _left_speed 	= 	speed->wheel_left;
  _right_speed 	= 	speed->wheel_right;
  vx = _wheel_radius*(_right_speed+_left_speed)/2.0;
  vy = 0.0;
  vth = _wheel_radius*(_right_speed - _left_speed)/(2.0*_wheel_base);
  cb_timer = ros::Time::now();

}

void resetOdomCb(const sensor_msgs::Joy::ConstPtr& msg) {
  if (msg->buttons[6] == 1){
    x=0.0;
    y=0.0;
    th=0.0;
  }

}

void imuCb(const sensor_msgs::Imu::ConstPtr& msg) {
    
  _orientation_data = msg->orientation;
  std::cout << "Received Imu!" << std::endl; // remove later 

  tf::Quaternion q(0,0,msg->orientation.z,msg->orientation.w);

  angular_velocity = msg->angular_velocity.z;


    // yaw (z-axis rotation)
    double siny_cosp = 2 * (msg->orientation.w * msg->orientation.z + msg->orientation.x * msg->orientation.y);
    double cosy_cosp = 1 - 2 * (msg->orientation.y * msg->orientation.y + msg->orientation.z * msg->orientation.z);
    theta = std::atan2(siny_cosp, cosy_cosp);

}

void getPara(ros::NodeHandle n_private) {
  n_private.param("odom_topic", _odom_topic_name, std::string("wheelodom"));
  n_private.param("broadcast_odom_tf", broadcast_odom_tf, true);
  n_private.param("speed_topic", _speed_pub_name, std::string("wheel_speed"));
  n_private.param("paw_frame", _paw_frame_name, std::string("base_link"));
  n_private.param("wheel_base", _wheel_base, double(0.39));
  n_private.param("wheel_radius", _wheel_radius, double(0.0625));
  n_private.param("loop_rate", _l_rate, int(50));
  n_private.param("speed_cb_timeout", cb_timeout, double(0.1));
  n_private.param("imu_topic", _imu_topic_name, std::string("/meerkat/vectornav/IMU"));

}


int main(int argc, char** argv) {

  ros::init(argc, argv, "odometry_node");
  ros::NodeHandle n;
  ros::NodeHandle n_p("~");
  getPara(n_p);
  ros::Publisher odom_pub = n.advertise<nav_msgs::Odometry>(_odom_topic_name, 1);
  ros::Subscriber _wheel_speed  = n.subscribe(_speed_pub_name, 1 , &speedCb);
  ros::Subscriber _reset_odom = n.subscribe("joy", 1 , &resetOdomCb);
  ros::Subscriber _imu_sub    = n.subscribe(_imu_topic_name, 1 , &imuCb);


  tf::TransformBroadcaster odom_broadcaster;

  ros::Time current_time, last_time;
  current_time = ros::Time::now();
  last_time = ros::Time::now();

  ros::Rate r(_l_rate);
  while(n.ok()){

    ros::spinOnce();               // check for incoming messages
    current_time = ros::Time::now();

    if ((current_time - cb_timer).toSec() < cb_timeout ) {

      double dt = (current_time - last_time).toSec();
      double delta_x = (vx * cos(th) - vy * sin(th)) * dt;
      double delta_y = (vx * sin(th) + vy * cos(th)) * dt;
      double delta_th = vth * dt;



      x += delta_x;
      y += delta_y;
      th += delta_th;

      //since all odometry is 6DOF we'll need a quaternion created from yaw
      geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(th);
      //geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(theta);




      if (broadcast_odom_tf){
        //first, we'll publish the transform over tf
        geometry_msgs::TransformStamped odom_trans;
        odom_trans.header.stamp = current_time;
        odom_trans.header.frame_id = "odom";
        odom_trans.child_frame_id = _paw_frame_name;

        odom_trans.transform.translation.x = x;
        odom_trans.transform.translation.y = y;
        odom_trans.transform.translation.z = 0.0;
        odom_trans.transform.rotation = odom_quat;

        //send the transform
        odom_broadcaster.sendTransform(odom_trans);
      }
      //next, we'll publish the odometry message over ROS
      nav_msgs::Odometry odom;
      odom.header.stamp = current_time;
      odom.header.frame_id = "odom";

      //set the position
      odom.pose.pose.position.x = x;
      odom.pose.pose.position.y = y;
      odom.pose.pose.position.z = 0.0;
      odom.pose.pose.orientation = odom_quat;

      //set the velocity
      odom.child_frame_id = _paw_frame_name;
      odom.twist.twist.linear.x = vx;
      odom.twist.twist.linear.y = vy;
      odom.twist.twist.angular.z = vth;
      // odom.twist.twist.angular.z = angular_velocity;

      //publish the message
      odom_pub.publish(odom);

    }
    else{
    //  ROS_INFO("Speed is not updated");
    }
    last_time = current_time;
    r.sleep();
  }
}
