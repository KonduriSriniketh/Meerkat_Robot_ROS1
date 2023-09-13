#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32
from serial import Serial, SerialException

def tof_publisher():
    # Initialize the ROS node
    rospy.init_node('tof_publisher', anonymous=True)

    # Define the serial port and baud rate for the Arduino
    serial_port = '/dev/ttyUSB1'  # Change this to your Arduino's serial port
    baud_rate = 9600  # Change this to match the Arduino's baud rate

    # Create a serial connection with the Arduino
    try:
        arduino_serial = Serial(serial_port, baud_rate)
    except SerialException as e:
        rospy.logerr(f"Error: {e}")
        return

    # Create a publisher for each TOF sensor data
    sensor_names = ['tof_sensor1', 'tof_sensor2', 'tof_sensor3', 'tof_sensor4', 'tof_sensor5', 'tof_sensor6']
    tof_pubs = {name: rospy.Publisher(name, Float32, queue_size=10) for name in sensor_names}

    while not rospy.is_shutdown():
        # Read data from the Arduino
        arduino_response = arduino_serial.readline().decode('latin-1').strip()
        #print(arduino_response)
        # Split the response into individual sensor data
        sensor_data = arduino_response.split()
        print(sensor_data)
        for data in sensor_data:
            try:
                sensor_id = int(sensor_data.index(data) + 1)
                data = float(data)
                print(sensor_id, data)
            except ValueError:
                rospy.logwarn("Invalid data received from Arduino.")
                continue

            # Publish the data to the corresponding topic
            topic_name = f'tof_sensor{sensor_id}'
            if topic_name in tof_pubs:
                tof_pubs[topic_name].publish(data)

    arduino_serial.close()

if __name__ == '__main__':
    try:
        tof_publisher()
    except rospy.ROSInterruptException:
        pass
