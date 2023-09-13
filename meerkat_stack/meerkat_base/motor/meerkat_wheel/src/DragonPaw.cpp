
#include "meerkat_wheel/DragonPaw.h"

void DragonPaw::cmdCb(const geometry_msgs::Twist &cmd_msg){
  _linear_x 	= 	cmd_msg.linear.x;
  _angular_z 	= 	cmd_msg.angular.z;
  _right_wheel_speed 	= 	(_linear_x / _wheel_radius) + ((_wheel_base/_wheel_radius) * _angular_z);
  _left_wheel_speed 	= 	(_linear_x / _wheel_radius) - ((_wheel_base/_wheel_radius) * _angular_z);
  if (_allow_cmd_timeout){
    _cb_timer = ros::Time::now();
  }
}


DragonPaw::DragonPaw(): nh_private("~")
{

  getParams();
  paw_controller = new Oriental_Driver();
  paw_controller->Connect(_paw_serial, _baud_rate, 2);
  _wheel_speed_pub = nh.advertise<meerkat_msgs::Speed>(_speed_pub_name,10);
  _battery_voltage_pub = nh.advertise<meerkat_msgs::Battery>(_voltage_pub_name,10);
  _cmd_sub = nh.subscribe(_cmd_sub_name, 1 , &DragonPaw::cmdCb, this);

  _x=0.0;
  _y=0.0;
  _th=0.0;


}

DragonPaw::~DragonPaw(){

  delete paw_controller;

}

bool DragonPaw::ReadFromPaw(int16_t &_left_wheel, int16_t &_right_wheel){

  int rd1_ = paw_controller->ReadSpeed(_left_motor_id,_left_wheel);
  int rd2_ = paw_controller->ReadSpeed(_right_motor_id,_right_wheel);

  paw_controller->ReadVoltage(_left_motor_id,_voltage_l);
  paw_controller->ReadVoltage(_right_motor_id,_voltage_r);
  _right_wheel = -1*_right_wheel;
  
  ROS_INFO("rd1 is %d", rd1_);
  ROS_INFO("rd2 is %d", rd2_);

  if (rd1_ && rd2_){
    //ROS_INFO("Read from Paw successful");
    return true;
  }
  else {
    ROS_WARN("Read from Paw ######### unsuccessful");
    return false;
  }
}

void DragonPaw::WriteToPaw(int r_w_speed, int l_w_speed){
  int rf1_, rf2_;
  //std::cout << r_w_speed <<" -- " << l_w_speed << '\n';
  if (_allow_cmd_timeout){
    const double _cb_dt = (ros::Time::now() - _cb_timer).toSec();

    if ( _cb_dt > _cb_timeout){
      //ROS_INFO("CallBack timeout detected!...Stopping the robot movement");
      stopPaw();
    }
    else{
      r_w_speed = -1*r_w_speed;
      rf1_ = paw_controller->WriteSpeed(_left_motor_id, l_w_speed);
      rf2_ = paw_controller->WriteSpeed(_right_motor_id, r_w_speed);
    }
  }
  else{
    r_w_speed = -1*r_w_speed;
   rf1_ = paw_controller->WriteSpeed(_left_motor_id, l_w_speed);
   rf2_ = paw_controller->WriteSpeed(_right_motor_id, r_w_speed);
  }
     //  if (rf1_ && rf2_ ){/*ROS_INFO("Write successful")*/ }
     // else{ROS_WARN("Write to Paw unsuccessfull");}
}
void DragonPaw::getParams(){

  nh_private.param("paw_port"           , _paw_serial           , std::string("/dev/ttyUSB1"));
  nh_private.param("baud_rate"           , _baud_rate           , int(115200));
  nh_private.param("speed_topic"         , _speed_pub_name      , std::string("wheel_speed"));
  nh_private.param("paw_frame"           , _paw_frame_name      , std::string("base_link"));
  nh_private.param("cmd_topic"  	       , _cmd_sub_name  	    , std::string("cmd_vel"));
  nh_private.param("voltage_topic"  	   , _voltage_pub_name  	, std::string("motor_voltage"));
  nh_private.param("loop_rate"           , _l_rate              , int(10));
  nh_private.param("wheel_base"		  		 , _wheel_base			    , double(0.39));
  nh_private.param("wheel_radius"		   	 , _wheel_radius			  , double(0.0625));
  nh_private.param("cb_timeout"					 , _cb_timeout				  , double(0.5));
  nh_private.param("allow_cmd_timeout"   , _allow_cmd_timeout   , true);
  nh_private.param("gear_ratio"           , _gear_ratio         , int(30));
  nh_private.param("paw_left_id"          , _left_motor_id      , int(1));
  nh_private.param("paw_right_id"        ,_right_motor_id       , int(2));
}

void DragonPaw::execute(){

  ros::Rate loop_rate(_l_rate);
  while	(ros::ok()){
    PublishWheelSpeed();
    PublishBatteryVoltage();
    ros::spinOnce();
    CheckForError();
    WriteToPaw(rpm_function(_right_wheel_speed), rpm_function(_left_wheel_speed));
    loop_rate.sleep();
  }
}


int DragonPaw::rpm_function(double value){

  _rpm = (value*_gear_ratio*60)/(2*pi);
  return _rpm;
}

double DragonPaw::rad_function(int16_t value){

  _rps = (value*2*pi)/(_gear_ratio*60);
  return _rps;
}

void DragonPaw::PublishWheelSpeed(){

  bool _read_flag 	= 		ReadFromPaw(l_wheel, r_wheel);

  if (_read_flag){
    speed.wheel_left  	= 	rad_function(l_wheel);
    speed.wheel_right 	= 	rad_function(r_wheel);
    _wheel_speed_pub.publish(speed);
  }
}

void DragonPaw::PublishBatteryVoltage(){

  meerkat_msgs::Battery battery;
  battery.voltage_left = _voltage_l;
  battery.voltage_right = _voltage_r;
  _battery_voltage_pub.publish(battery);

}


void DragonPaw::breakPaw() {
  paw_controller->BreakAll();
}
void DragonPaw::stopPaw() {
  paw_controller->StopAll();
}

void DragonPaw::CheckForError() {

  paw_controller->CheckAlarm(_left_motor_id, _error_left);
  paw_controller->CheckAlarm(_right_motor_id, _error_right);

  switch (_error_left) {
    case Oriental_Driver::CPU_ERROR: ROS_FATAL("CPU_ERROR detected in left motor!!"); break;
    case Oriental_Driver::EPPROM_ERROR: ROS_FATAL("EPPROM_ERROR detected in left motor!!"); break;
    case Oriental_Driver::OVER_CURRENT: ROS_FATAL("OVER_CURRENT detected in left motor!!"); break;
    case Oriental_Driver::OPERATION_PREVENTION_ERROR: ROS_ERROR("OPERATION_PREVENTION_ERROR detected in left motor!!"); break;
    case Oriental_Driver::COMMUNICATION_ERROR: ROS_ERROR("COMMUNICATION_ERROR detected in left motor!!"); break;
    case Oriental_Driver::COMMUNICATION_TIMEOUT_ERROR: ROS_ERROR("COMMUNICATION_TIMEOUT_ERROR detected in left motor!!"); break;
    case Oriental_Driver::SENSOR_ERROR_AT_POWER_ON: ROS_ERROR("SENSOR_ERROR_AT_POWER_ON detected in left motor!!"); break;
    case Oriental_Driver::MAIN_CIRCUIT_OVERHEAT: ROS_ERROR("MAIN_CIRCUIT_OVERHEAT detected in left motor!!"); break;
    case Oriental_Driver::OVER_VOLTAGE: ROS_ERROR("OVER_VOLTAGE detected in left motor!!"); break;
    case Oriental_Driver::UNDER_VOLTAGE: ROS_ERROR("UNDER_VOLTAGE detected in left motor!!"); break;
    case Oriental_Driver::SENSOR_ERROR: ROS_ERROR("SENSOR_ERROR detected in left motor!!"); break;
    case Oriental_Driver::OVERLOAD: ROS_ERROR("OVERLOAD detected in left motor!!"); break;
    case Oriental_Driver::OVERSPEED: ROS_ERROR("OVERSPEED detected in left motor!!"); break;
  }
  switch (_error_right) {
    case Oriental_Driver::CPU_ERROR: ROS_FATAL("CPU_ERROR detected in right motor!!"); break;
    case Oriental_Driver::EPPROM_ERROR: ROS_FATAL("EPPROM_ERROR detected in right motor!!"); break;
    case Oriental_Driver::OVER_CURRENT: ROS_FATAL("OVER_CURRENT detected in right motor!!"); break;
    case Oriental_Driver::OPERATION_PREVENTION_ERROR: ROS_ERROR("OPERATION_PREVENTION_ERROR detected in right motor!!"); break;
    case Oriental_Driver::COMMUNICATION_ERROR: ROS_ERROR("COMMUNICATION_ERROR detected in right motor!!"); break;
    case Oriental_Driver::COMMUNICATION_TIMEOUT_ERROR: ROS_ERROR("COMMUNICATION_TIMEOUT_ERROR detected in right motor!!"); break;
    case Oriental_Driver::SENSOR_ERROR_AT_POWER_ON: ROS_ERROR("SENSOR_ERROR_AT_POWER_ON detected in right motor!!"); break;
    case Oriental_Driver::MAIN_CIRCUIT_OVERHEAT: ROS_ERROR("MAIN_CIRCUIT_OVERHEAT detected in right motor!!"); break;
    case Oriental_Driver::OVER_VOLTAGE: ROS_ERROR("OVER_VOLTAGE detected in right motor!!"); break;
    case Oriental_Driver::UNDER_VOLTAGE: ROS_ERROR("UNDER_VOLTAGE detected in right motor!!"); break;
    case Oriental_Driver::SENSOR_ERROR: ROS_ERROR("SENSOR_ERROR detected in right motor!!"); break;
    case Oriental_Driver::OVERLOAD: ROS_ERROR("OVERLOAD detected in right motor!!"); break;
    case Oriental_Driver::OVERSPEED: ROS_ERROR("OVERSPEED detected in right motor!!"); break;
  }



}
