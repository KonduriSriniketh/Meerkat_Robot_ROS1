#!/usr/bin/env python

import rospy
from maps.srv import GenerateImage
import subprocess

def handle_service_request(request):
    # Execute the external script and capture the output (image or image path)
    script_path = "/home/sutd/catkin_ws/src/maps/src/heatmap_generation.py"
    output = subprocess.check_output(["python", script_path])

    # Set the image path (or output) in the service response
    response = GenerateImageResponse()
    response.image_path = output.decode('utf-8')  # Convert bytes to string
    return response

def generate_image_server():
    rospy.init_node('generate_image_server')
    rospy.Service('generate_image', GenerateImage, handle_service_request)
    rospy.spin()

if __name__ == '__main__':
    generate_image_server()

