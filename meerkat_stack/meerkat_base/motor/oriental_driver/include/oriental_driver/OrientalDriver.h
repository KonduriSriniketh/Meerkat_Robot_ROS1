

#ifndef ORIENTAL_DRIVER_H_
#define ORIENTAL_DRIVER_H_


#include <iostream>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <serial/serial.h>
#include <math.h>
#include <string>


class Oriental_Driver{

private:

  serial::Serial * port_;
  std::string _port_name;
  int _baud_rate;
  int _parity_bit;

  bool debug_;
  int current_slave_;
  int fb_, rd_;
  size_t bytes_wrote;
  double _volt;
  const static uint8_t table_crc_hi[];
  const static uint8_t table_crc_lo[];

  int read_single(int function, int addr, int nb, uint16_t *dest);
  uint16_t crc16(uint8_t *buff, uint16_t buffer_length);
  int crc_update(uint8_t *req, int req_length);
  int send_msg(uint8_t *msg, int msg_length, int nb);
  int rtu_build_request(int function, int addr, int nb, uint8_t *req);
  int write_single(int function, int addr, const uint16_t value);
  int read_register(int slave, int addr, int nb, uint16_t *dest);
  int write_register(int slave, int addr, const uint16_t value);
  bool OpenPort();



public:

  Oriental_Driver();
  ~Oriental_Driver();
  bool Connect(std::string port, int baud, int parity);
  bool WriteSpeed(int slave, int speed);
  int ReadSpeed(int slave, int16_t &speed);
  int ReadVoltage(int slave, double &voltage);
  int BreakAll();
  int StopAll();
  int Stop(int slave);
  void CheckAlarm(int slave, int16_t &error);
  void ClearAlarm(int slave);


  enum {
      WRITE_SINGLE_REGISTER = 0x06,
      READ_SINGLE_REGISTER = 0x03,
      WRITE_MULTIPLE_REGISTER = 0x10,
      READ_WRITE_MULTIPLE_REGISTER = 0x17,
      DIAGNOSIS_REGISTER = 0x08,


      OPERATION_REGISTER_0 = 1153, // upper register 1152 and lower register 1153 -- 1152 is 00h
      OPERATION_REGISTER_1 = 1155, // upper register 1154 and lower register 1155 -- 1154 is 00h
      OPERATION_REGISTER_2 = 1157,
      OPERATION_REGISTER_3 = 1159,
      OPERATION_REGISTER_4 = 1161,
      OPERATION_REGISTER_5 = 1163,
      OPERATION_REGISTER_6 = 1165,
      OPERATION_REGISTER_7 = 1167,

      SPEED_READ_REGISTER = 206, // upper register 206 and lower register 207
      LOAD_READ_REGISTER = 216,  // upper register 216 and lower register 217
      TEMPERATURE_READ_REGISTER = 248, // upper register 248 and lower register 249
      INVERTER_VOLATGE_READ_REGISTER = 326, // upper register 326 and lower register 327
      ALARM_RESET_REGISTER = 384, // upper register 384 and lower register 385
      WRITE_SPEED_REGISTER = 125, //upper register 124 and lower register 125 -- 124 is 00h
      ALARM_READ_REGISTER = 128, // upper register 128 and lower register 129

      WRITE_SINGLE_RESPONSE_LENGTH = 8,
      MAX_READ_REGISTERS = 16,
      RTU_PRESET_LENGTH = 6,
      MAX_MESSAGE_LENGTH = 100,
      MIN_MESSAGE_LENGTH = 12,

      /*  control data for OPERATION_REGISTER_0 */
      BRAKE = 40,
      STOP = 48,
      MOTOR_FWD = 56,
      MOTOR_REV = 24,

      /*  control data for OPERATION_REGISTER_1  -- with min speed*/
      MOTOR_FWD_1 = 57,
      MOTOR_REV_1 = 25,
      STOP_1 = 49,
      BRAKE_1 = 41,


      CLEAR_ALARM = 128,

      MIN_SPEED = 80,
      MAX_SPEED = 3150,
      TORQUE_LIMITING_MAX = 200,
      ACCEL_MAX = 150,
      ACCEL_DEFAULT = 5,
      ACCEL_MIN = 1, // 1 = 0.1s
      DEACCEL_MAX = 150,
      DEACCEL_DEFAULT = 5,
      DEACCEL_MIN = 1, // 1 = 0.1s

      SILENT_INTERVAL = 3500, //3.5ms
      TRANSMISSION_WAITING_TIME = 12000, // 10ms

      /* error codes -- Possible to reset */

      OPERATION_PREVENTION_ERROR = 70, // 46h
      COMMUNICATION_ERROR = 132, //84h
      COMMUNICATION_TIMEOUT_ERROR = 133, //85h
      SENSOR_ERROR_AT_POWER_ON = 66, //42h
      MAIN_CIRCUIT_OVERHEAT = 33,//21h
      OVER_VOLTAGE = 34,//22h
      UNDER_VOLTAGE = 37,//25h
      SENSOR_ERROR = 40,//28h
      OVERLOAD = 48,//30h
      OVERSPEED = 49,//31h


      /* error codes -- Not possible to reset*/
      CPU_ERROR = 240, //F0h
      EPPROM_ERROR = 65, //41h
      OVER_CURRENT = 32, //20h

  };

};








#endif
