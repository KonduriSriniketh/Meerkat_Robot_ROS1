/*
* file: DragonPaw.h
* Authors: Sriharsha Ghanta
*
* Oceania Robotics Pte Ltd
* PRIVATE AND CONFIDENTIAL
* ----------------------------------------------------------
*  Oceania Robotics Pte Ltd
*  All Rights Reserved.
*
*  NOTICE:  All information contained herein is, and remains
*  the property of Oceania Robotics Pte Ltd.
* -----------------------------------------------------------
*/
#ifndef DF_CLAW_H
#define DF_CLAW_H

#include "oriental_driver/OrientalDriver.h"
#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include "std_msgs/Int32.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/Quaternion.h"
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/Joy.h>
#include <sensor_msgs/JoyFeedbackArray.h>
#include "meerkat_msgs/Speed.h"
#include "meerkat_msgs/Battery.h"

#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <math.h>
#include <float.h>



class DragonPaw{

private:

	ros::NodeHandle nh;
	ros::NodeHandle nh_private;

	ros::Publisher _wheel_speed_pub;
	ros::Publisher _battery_voltage_pub;
	ros::Subscriber _cmd_sub;
	ros::Subscriber _joystick;
	
	ros::Time stamp;
	meerkat_msgs::Speed speed;

	Oriental_Driver *paw_controller;
	const double pi = boost::math::constants::pi<double>();

	void getParams();
	void cmdCb(const geometry_msgs::Twist &cmd_msg);
	bool ReadFromPaw(int16_t &_left_wheel, int16_t &_right_wheel);
	void WriteToPaw(int r_w_speed, int l_w_speed);
	int rpm_function(double value);
	double rad_function(int16_t value);
	void PublishWheelSpeed();
	void PublishBatteryVoltage();
	void stopPaw();
	void breakPaw();
	void CheckForError();
	double _linear_x{0.0};
	double _angular_z{0.0};
	double _right_wheel_speed{0.0};
	double _left_wheel_speed{0.0};

	int _rpm;
	double _rps;
	int _right_QPPS;
	int _left_QPPS;
	double _left_speed;
	double _right_speed;
	double _vx;
	double _vy;
	double _vth;
	double _delta_x;
	double _delta_y;
	double _delta_th;
	double _x;
	double _y;
	double _z;
	double _th;

	int _gear_ratio;
	int16_t l_wheel{0};
	int16_t r_wheel{0};
	int16_t _error_left{0};
	int16_t _error_right{0};
	double _dt;

	std::string _paw_serial;
	std::string _speed_pub_name;
	std::string _voltage_pub_name;
	std::string _paw_frame_name;
	std::string _cmd_sub_name;
	std::string _reset_odom_sub_name;

	int _time_out;
	int _baud_rate;
	int _l_rate;
  	int _right_motor_id;
	int _left_motor_id;
	double _wheel_base;
	double _wheel_radius;
	int _pulse_per_revolution;
	ros::Time _cb_timer;
  	ros::Time _curr_time;
	ros::Time _prev_time;
	double _cb_timeout;
  	bool _allow_cmd_timeout;
	double _voltage_l{0};
	double _voltage_r{0};

public:
	DragonPaw();
	~DragonPaw();
	void execute();

};





#endif
