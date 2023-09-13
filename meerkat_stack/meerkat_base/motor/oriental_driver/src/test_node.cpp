
#include <ros/ros.h>
#include <oriental_driver/OrientalDriver.h>
#include <unistd.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "oriental_driver_test_node");

  Oriental_Driver modbus_driver_;
  while(1){
    if (modbus_driver_.Connect("/dev/ttyUSB0", 115200, 2)) break;;

  }

  ros::NodeHandle nh;

  ros::Rate rate(50);
  // modbus_driver_.WriteSpeed(2, 2000);
  // for (int i = 81; i < 3125; i=i+5){
  //   modbus_driver_.WriteSpeed(1, i);
  //   modbus_driver_.WriteSpeed(2, i);
  //
  // }

  int16_t l_wheel_speed;
  int16_t r_wheel_speed;
  int16_t error1;
  int16_t error2;

  while (ros::ok()) {

    ros::Time start_time = ros::Time::now();
    double volt;
     modbus_driver_.ReadVoltage(1, volt);
    // std::cout << "Voltage1: " << volt << '\n';
    // modbus_driver_.ReadVoltage(2, volt);
    // std::cout << "Voltage2: " << volt << '\n';
    if (volt > 0){
    modbus_driver_.WriteSpeed(1, 300);
    modbus_driver_.WriteSpeed(2, 300);

    std::this_thread::sleep_for(std::chrono::microseconds(5*1000000));
     modbus_driver_.ReadSpeed(1, l_wheel_speed);
     modbus_driver_.ReadSpeed(2, r_wheel_speed);
     std::cout << "r_wheel_speed1: " << l_wheel_speed<<'\n';
     std::cout << "r_wheel_speed2: " << r_wheel_speed<<'\n';

     modbus_driver_.WriteSpeed(1, -300);
     modbus_driver_.WriteSpeed(2, -300);

     std::this_thread::sleep_for(std::chrono::microseconds(5*1000000));
      modbus_driver_.ReadSpeed(1, l_wheel_speed);
      modbus_driver_.ReadSpeed(2, r_wheel_speed);
      std::cout << "r_wheel_speed1: " << l_wheel_speed<<'\n';
      std::cout << "r_wheel_speed2: " << r_wheel_speed<<'\n';
}
    modbus_driver_.CheckAlarm(1,error1);
    std::cout << "Error from 1: " << error1 << '\n';
    modbus_driver_.CheckAlarm(2,error2);
    std::cout << "Error from 2: " << error2 << '\n';

    // if (error1 > 0){
    //   modbus_driver_.ClearAlarm(1);
    // }
    // if (error2 > 0){
    //   modbus_driver_.ClearAlarm(2);
    // }

     //modbus_driver_.StopAll();
   //std::this_thread::sleep_for(std::chrono::microseconds(3*1000000));

     // modbus_driver_.ReadSpeed(1, r_wheel_speed);

    rate.sleep();
  }


  return 0;
}
