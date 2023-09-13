#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from meerkat_msgs.msg import Battery

height = 300
width  = 500
font = cv2.FONT_HERSHEY_SIMPLEX
blank_image = np.zeros((height,width,3), np.uint8)
cv2.putText(blank_image, 'Voltage L', (10,40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(blank_image, 'Voltage R', (10,80), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
def callback_number(msg):
    cv2.putText(blank_image, 'Christmas', (10,10), font, 4, (0, 255, 0), 2, cv2.LINE_AA)



# The function cv2.imshow() is used to display an image in a window.
cv2.imshow('graycsale image',blank_image)

# waitKey() waits for a key press to close the window and 0 specifies indefinite loop
cv2.waitKey(0)

# cv2.destroyAllWindows() simply destroys all the windows we created.
cv2.destroyAllWindows()


