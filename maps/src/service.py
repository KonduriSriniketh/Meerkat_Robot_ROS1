#!/usr/bin/env python3

import rospy
from maps.srv import GenerateImage
from maps.msg import GenerateImageResponse
import subprocess

def handle_service_request(request):
    # Execute the external script and capture the output (image or image path)
    script_path = "/home/sutd/catkin_ws/src/maps/src/heatmap_generation.py"
    output = subprocess.Popen(["python3", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Set the image path (or output) in the service response
    response = GenerateImageResponse()
    response.success = True
    response.message = "Script Done!"
    return response

def generate_image_server():
    rospy.init_node('generate_image_server')
    rospy.Service('generate_image', GenerateImage, handle_service_request)
    rospy.spin()

if __name__ == '__main__':
    generate_image_server()

