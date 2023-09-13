
#include "oriental_driver/OrientalDriver.h"

const uint8_t Oriental_Driver::table_crc_hi[] = {
  0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0,
  0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
  0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
  0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
  0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1,
  0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
  0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1,
  0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
  0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0,
  0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40,
  0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1,
  0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
  0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0,
  0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40,
  0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
  0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
  0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0,
  0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
  0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
  0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
  0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
  0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40,
  0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1,
  0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
  0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0,
  0x80, 0x41, 0x00, 0xC1, 0x81, 0x40
};

const uint8_t Oriental_Driver::table_crc_lo[] = {
  0x00, 0xC0, 0xC1, 0x01, 0xC3, 0x03, 0x02, 0xC2, 0xC6, 0x06,
  0x07, 0xC7, 0x05, 0xC5, 0xC4, 0x04, 0xCC, 0x0C, 0x0D, 0xCD,
  0x0F, 0xCF, 0xCE, 0x0E, 0x0A, 0xCA, 0xCB, 0x0B, 0xC9, 0x09,
  0x08, 0xC8, 0xD8, 0x18, 0x19, 0xD9, 0x1B, 0xDB, 0xDA, 0x1A,
  0x1E, 0xDE, 0xDF, 0x1F, 0xDD, 0x1D, 0x1C, 0xDC, 0x14, 0xD4,
  0xD5, 0x15, 0xD7, 0x17, 0x16, 0xD6, 0xD2, 0x12, 0x13, 0xD3,
  0x11, 0xD1, 0xD0, 0x10, 0xF0, 0x30, 0x31, 0xF1, 0x33, 0xF3,
  0xF2, 0x32, 0x36, 0xF6, 0xF7, 0x37, 0xF5, 0x35, 0x34, 0xF4,
  0x3C, 0xFC, 0xFD, 0x3D, 0xFF, 0x3F, 0x3E, 0xFE, 0xFA, 0x3A,
  0x3B, 0xFB, 0x39, 0xF9, 0xF8, 0x38, 0x28, 0xE8, 0xE9, 0x29,
  0xEB, 0x2B, 0x2A, 0xEA, 0xEE, 0x2E, 0x2F, 0xEF, 0x2D, 0xED,
  0xEC, 0x2C, 0xE4, 0x24, 0x25, 0xE5, 0x27, 0xE7, 0xE6, 0x26,
  0x22, 0xE2, 0xE3, 0x23, 0xE1, 0x21, 0x20, 0xE0, 0xA0, 0x60,
  0x61, 0xA1, 0x63, 0xA3, 0xA2, 0x62, 0x66, 0xA6, 0xA7, 0x67,
  0xA5, 0x65, 0x64, 0xA4, 0x6C, 0xAC, 0xAD, 0x6D, 0xAF, 0x6F,
  0x6E, 0xAE, 0xAA, 0x6A, 0x6B, 0xAB, 0x69, 0xA9, 0xA8, 0x68,
  0x78, 0xB8, 0xB9, 0x79, 0xBB, 0x7B, 0x7A, 0xBA, 0xBE, 0x7E,
  0x7F, 0xBF, 0x7D, 0xBD, 0xBC, 0x7C, 0xB4, 0x74, 0x75, 0xB5,
  0x77, 0xB7, 0xB6, 0x76, 0x72, 0xB2, 0xB3, 0x73, 0xB1, 0x71,
  0x70, 0xB0, 0x50, 0x90, 0x91, 0x51, 0x93, 0x53, 0x52, 0x92,
  0x96, 0x56, 0x57, 0x97, 0x55, 0x95, 0x94, 0x54, 0x9C, 0x5C,
  0x5D, 0x9D, 0x5F, 0x9F, 0x9E, 0x5E, 0x5A, 0x9A, 0x9B, 0x5B,
  0x99, 0x59, 0x58, 0x98, 0x88, 0x48, 0x49, 0x89, 0x4B, 0x8B,
  0x8A, 0x4A, 0x4E, 0x8E, 0x8F, 0x4F, 0x8D, 0x4D, 0x4C, 0x8C,
  0x44, 0x84, 0x85, 0x45, 0x87, 0x47, 0x46, 0x86, 0x82, 0x42,
  0x43, 0x83, 0x41, 0x81, 0x80, 0x40
};

Oriental_Driver::Oriental_Driver(): debug_{false}, current_slave_{0}{

  port_ = new serial::Serial();
}

Oriental_Driver::~Oriental_Driver(){

  write_register(0, WRITE_SPEED_REGISTER, BRAKE);
  port_->close();
  delete port_;

}

bool Oriental_Driver::Connect(std::string port, int baud, int parity){

  _port_name = port;
  _baud_rate = baud;
  _parity_bit = parity;

  port_->setPort(_port_name);
  port_->setBaudrate(_baud_rate);

  if (_parity_bit == 0) port_->setParity(serial::parity_t::parity_none);
  else if (_parity_bit == 1) port_->setParity(serial::parity_t::parity_odd);
  else if (_parity_bit == 2) port_->setParity(serial::parity_t::parity_even);
  else throw std::invalid_argument("Parity input is not valid");

  serial::Timeout _time_out = serial::Timeout::simpleTimeout(10);
  port_->setTimeout(_time_out);
  OpenPort();
  //  write_register(slave, WRITE_SPEED_REGISTER, CLEAR_ALARM);
}

bool Oriental_Driver::OpenPort(){
  int count = 1 ;
  while (count > 0){
    try{
      port_->open();
      if (port_->isOpen()) return true;
    }
    catch (serial::IOException& e){
      std::cout << "Waiting for the port... Please check the connection!"<< std::endl;
      usleep(1000*500);
      count ++;
      if (count > 30){
        std::cout <<"Port opening timeout detected... Exiting the program!" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    catch (serial::SerialException& e){
      if (port_->isOpen()) return true;
    }
  }

}

void Oriental_Driver::CheckAlarm(int slave, int16_t &error){
  uint16_t data[2]= {};
  read_register(slave, ALARM_READ_REGISTER, 2, data);
  error = data[1];
  switch (error) {
    case OPERATION_PREVENTION_ERROR:
      ClearAlarm(slave); break;
    case COMMUNICATION_ERROR:
      std::cout << "COMMUNICATION_ERROR detected!! -- Clearing..." << '\n';
      ClearAlarm(slave); break;
    case SENSOR_ERROR_AT_POWER_ON:
      std::cout << "SENSOR_ERROR_AT_POWER_ON detected!! -- Clearing" << '\n';
      ClearAlarm(slave);
    case COMMUNICATION_TIMEOUT_ERROR:
      std::cout << "COMMUNICATION_TIMEOUT_ERROR detected!! -- Clearing..." << '\n';
      ClearAlarm(slave); break;
    case MAIN_CIRCUIT_OVERHEAT:
      std::cout << "MAIN_CIRCUIT_OVERHEAT detected!! -- Clearing..." << '\n';
      ClearAlarm(slave); break;
    case OVER_VOLTAGE:
      std::cout << "OVER_VOLTAGE detected!! -- Clearing..." << '\n';
      ClearAlarm(slave); break;
    case UNDER_VOLTAGE:
      std::cout << "UNDER_VOLTAGE detected!! -- Clearing..." << '\n';
      ClearAlarm(slave); break;
    case SENSOR_ERROR:
      std::cout << "SENSOR_ERROR detected!! --Clearing..."<< '\n';
      ClearAlarm(slave); break;
    case OVERLOAD:
      std::cout << "OVERLOAD detected!! -- Clearing..."<< '\n';
      ClearAlarm(slave); break;
    case OVERSPEED:
      std::cout << "OVERSPEED detected!! -- Clearing..."<< '\n';
      ClearAlarm(slave); break;
    case EPPROM_ERROR:
      std::cout << "EPPROM_ERROR detected!! -- Cannot clear this error!!" << '\n';
      exit(EXIT_FAILURE);
    case CPU_ERROR:
      std::cout << "CPU_ERROR detected!! -- Cannot clear this error!!"<< '\n';
      exit(EXIT_FAILURE);
    case OVER_CURRENT:
      std::cout << "OVER_CURRENT detected!! -- Cannot clear this error!!"<< '\n';
      exit(EXIT_FAILURE);
  }
}

void Oriental_Driver::ClearAlarm(int slave) {
  write_register(slave, WRITE_SPEED_REGISTER, CLEAR_ALARM);
}
bool Oriental_Driver::WriteSpeed(int slave, int speed){

  //std::cout << "sPEED from driver: " << speed << '\n';

  if ((abs(speed) > MIN_SPEED) && (abs(speed) < MAX_SPEED)){
    write_register(slave, OPERATION_REGISTER_0, (uint16_t)abs(speed));
    if (speed > 0) write_register(slave, WRITE_SPEED_REGISTER, MOTOR_FWD);
    else write_register(slave, WRITE_SPEED_REGISTER, MOTOR_REV);
    return true;
  }
  else if (speed == 0){
    write_register(slave, WRITE_SPEED_REGISTER, STOP);
  }
  else {
    //std::cout << "velocity not accepted" << '\n';
    return false;
  }

}

int Oriental_Driver::ReadSpeed(int slave, int16_t &speed){


  uint16_t speed_data[2] = {};
  int fb_ = read_register(slave, SPEED_READ_REGISTER, 2, speed_data);
  speed = speed_data[1];
  return fb_;

}

int Oriental_Driver::ReadVoltage(int slave, double &voltage){


  uint16_t voltage_data[2] = {};
  int fb_ = read_register(slave, INVERTER_VOLATGE_READ_REGISTER, 2, voltage_data);
  voltage = ((double)voltage_data[1]*0.1);
  return fb_;

}

int Oriental_Driver::BreakAll() {

  current_slave_ = 0;
  return write_register(0, WRITE_SPEED_REGISTER, BRAKE);

}

int Oriental_Driver::StopAll(){

  current_slave_ = 0;
  return write_register(0, WRITE_SPEED_REGISTER, STOP);

}

int Oriental_Driver::Stop(int slave) {

  return write_register(slave, WRITE_SPEED_REGISTER, STOP);

}

int Oriental_Driver::write_register(int slave, int addr, const uint16_t value){
  try{
    current_slave_ = slave;
    return write_single(WRITE_SINGLE_REGISTER, addr, value);
  }
  catch(const char* a ){
    //std::cout << a << '\n';
    return -1;
  }
}

int Oriental_Driver::read_register(int slave, int addr, int nb, uint16_t *dest){
  try{
    current_slave_ = slave;
    return read_single(READ_SINGLE_REGISTER,addr, nb, dest);
  }
  catch (const char* a){
    //std::cout << a << '\n';
  }
}

int Oriental_Driver::write_single(int function, int addr, const uint16_t value){

  int req_length;
  uint8_t req[MIN_MESSAGE_LENGTH];

  req_length = rtu_build_request(function, addr, (int) value, req);

  int bytes_wrote = send_msg(req, req_length, 0);

  if (current_slave_ !=0){

    uint8_t buffer[WRITE_SINGLE_RESPONSE_LENGTH];

    if (OpenPort()){
      fb_ = port_->read(buffer, WRITE_SINGLE_RESPONSE_LENGTH);
    }
    if(fb_!=WRITE_SINGLE_RESPONSE_LENGTH) throw "Nothing is read";

    if (debug_) {
      std::cout << "Reading..." << '\n';
      for (auto i = 0; i < WRITE_SINGLE_RESPONSE_LENGTH; i++) {
        printf("[%.2X]", buffer[i]);
      }
      printf("\n");
    }
    uint8_t rd_buffer[fb_-2];

    for(int i=0; i<fb_-2; i++){ rd_buffer[i] = buffer[i];  }

    crc_update(rd_buffer,fb_-2);

    if((((uint16_t)buffer[WRITE_SINGLE_RESPONSE_LENGTH-2] << 8) | buffer[WRITE_SINGLE_RESPONSE_LENGTH-1])
    ==(((uint16_t)rd_buffer[fb_-2] << 8) | rd_buffer[fb_-1])){
      return 1;
    }
    else throw "CRC Check failed";
  }
}

int Oriental_Driver::rtu_build_request(int function, int addr, int nb, uint8_t *req){

  req[0] = current_slave_;
  req[1] = function;
  req[2] = addr >> 8;
  req[3] = addr & 0x00ff;
  req[4] = nb >> 8;
  req[5] = nb & 0x00ff;

  return RTU_PRESET_LENGTH;
}

/* Sends a request/response */
int Oriental_Driver::send_msg(uint8_t *msg, int msg_length, int nb){

  msg_length = crc_update(msg, msg_length);

  std::this_thread::sleep_for(std::chrono::microseconds(SILENT_INTERVAL));
  if (OpenPort()) {
    bytes_wrote = port_->write(msg, msg_length);
  }

  std::this_thread::sleep_for(std::chrono::microseconds(SILENT_INTERVAL + TRANSMISSION_WAITING_TIME));

  if (debug_) {
    std::cout << "Writing..." << '\n';
    for (auto i = 0; i < msg_length; i++) {
      printf("[%.2X]", msg[i]);
    }

    printf("\n");
  }
  if (bytes_wrote == msg_length)  return 1;
  else return -1;
}

int Oriental_Driver::crc_update(uint8_t *req, int req_length){

  uint16_t crc = crc16(req, req_length);
  req[req_length++] = crc >> 8;
  req[req_length++] = crc & 0x00FF;

  return req_length;
}

uint16_t Oriental_Driver::crc16(uint8_t *buff, uint16_t buffer_length){

  uint8_t crc_hi = 0xFF; /* high CRC byte initialized */
  uint8_t crc_lo = 0xFF; /* low CRC byte initialized */
  unsigned int i; /* will index into CRC lookup */

  /* pass through message buffer */
  while (buffer_length--) {
    i = crc_hi ^ *buff++; /* calculate the CRC  */
    crc_hi = crc_lo ^ table_crc_hi[i];
    crc_lo = table_crc_lo[i];
  }

  return (crc_hi << 8 | crc_lo);
}

/* Reads the data from a remove device and put that data into an array */
int Oriental_Driver::read_single(int function, int addr, int nb, uint16_t *dest) {

  int rc;
  int req_length;
  uint8_t req[MIN_MESSAGE_LENGTH];
  uint8_t rsp[MAX_MESSAGE_LENGTH];

  if (nb > MAX_READ_REGISTERS) {
    // TO DO : Handle error
    return -1;
  }

  req_length = rtu_build_request(function, addr, nb, req);

  rc = send_msg(req, req_length, nb);

  int msg_length = 5 + (2*nb);

  uint8_t buffer[msg_length];
  if (OpenPort()){
    rd_ = port_->read(buffer, msg_length);
  }
  // TO DO :after reading, need to check crc
  // try/catch error
  if (rd_ != msg_length) throw "Nothing read";
  if (debug_) {
    std::cout << "Reading..." << '\n';

    for (auto i = 0; i < msg_length; i++) {
      printf("[%.2X]", buffer[i]);
    }

    printf("\n");
  }
  uint8_t rd_buffer[rd_-2];

  for(int i=0; i<rd_-2; i++){ rd_buffer[i] = buffer[i];  }

  crc_update(rd_buffer,rd_-2);
  if((((uint16_t)buffer[msg_length-2] << 8) | buffer[msg_length-1])
  ==(((uint16_t)rd_buffer[rd_-2] << 8) | rd_buffer[rd_-1])){
    for (auto i = 0; i < nb; i++) {
      dest[i] = ((uint16_t)buffer[3+(i*2)] << 8) | buffer[3+(i*2)+1];
    }
    return 1;}
    else throw "CRC Check failed";



  }
