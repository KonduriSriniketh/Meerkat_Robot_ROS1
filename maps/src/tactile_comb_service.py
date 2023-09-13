#!/usr/bin/env python3

import rospy
from maps.srv import GenerateImage
from maps.msg import GenerateImageResponse
import subprocess

def handle_service_request(request):
    # Execute the external script and capture the output (image or image path)
    script_path = "/home/sutd/meerkat_ros_ws/src/maps/src/tactile_comb_deployment.py"
    output = subprocess.Popen(["python3", script_path])

    # Set the image path (or output) in the service response
    response = GenerateImageResponse()
    response.success = True
    response.message = "Script Done!"
    return response

def generate_image_server():
    rospy.init_node('tactile_comb_server')
    rospy.Service('tactile_comb', GenerateImage, handle_service_request)
    rospy.spin()

if __name__ == '__main__':
    generate_image_server()

