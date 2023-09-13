#!/usr/bin/env python3

import cv2
import pandas as pd
import time
import numpy as np
import tkinter as tk
from tkinter import filedialog

class getPoints:

    def __init__(self, image_path):

        self.path = image_path
        self.points_x = []
        self.points_y = []
        self.den_vals = []

    def __del__(self):

        print('Destructor callefd')

    def click_event(self, event, x, y, flags, params):
    	    	
        if event == cv2.EVENT_LBUTTONDOWN:

            # y_new = self.img.shape[1] - y
            y_new = y
            x_new = x
            self.points_x.append(x_new)
            self.points_y.append(y_new)
            print('You choose-> ', x, ' ', y_new)
            value = input('Enter value: ')
            self.den_vals.append(value)
            self.img = cv2.circle(self.img, (x,y), 5, (255, 0, 0), 2)
            cv2.imshow('image', self.img)
     
        if event==cv2.EVENT_RBUTTONDOWN:
            print("Please use right button")

    def drive_gui(self):

        splash_img = np.ones((380,500,3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        splash_img = cv2.putText(splash_img, "Welcome", (100,100), font, 2, (255, 0, 0), 2)
        splash_img = cv2.putText(splash_img, "This Program is created by Thejus.",              (10,200), font, 0.5, (255, 0, 0), 1)
        splash_img = cv2.putText(splash_img, "Follow the instructions please",                  (10,220), font, 0.5, (255, 255, 0), 1)
        splash_img = cv2.putText(splash_img, "[*] Click on the map to record a point.",         (10,240), font, 0.5, (255, 255, 0), 1)
        splash_img = cv2.putText(splash_img, "[*] Hit 'a' to save the points at any time.",     (10,260), font, 0.5, (255, 255, 0), 1)
        splash_img = cv2.putText(splash_img, "[*] Hit 'Esc' to exit at any time.",              (10,280), font, 0.5, (255, 255, 0), 1)

        splash_img = cv2.putText(splash_img, "If instructions are clear... .",                  (5,320), font, 0.8, (255, 0, 255), 1)
        splash_img = cv2.putText(splash_img, "Press any Key to continue",                       (10,350), font, 1, (255, 0, 255), 1)
        cv2.imshow('Welcome', splash_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        root = tk.Tk()
        root.withdraw()
        self.image_path = filedialog.askopenfilename()
        self.img = cv2.imread(self.image_path, 1)

        cv2.imshow('image', self.img)
        cv2.setMouseCallback('image', self.click_event)

        byebye = False

        while not byebye:

            try:

                k = cv2.waitKey(0)
                if k==27:
                    print('Bye bye!')
                    byebye = True
                    cv2.destroyAllWindows()
                elif k== 97:
                    print('Saving the points.. press esc to quit')
                    out_df = pd.DataFrame(list(zip(self.points_x, self.points_y, self.den_vals)), columns =['x_points', 'y_points', 'density'])
                    out_df.to_csv('/home/sutd/catkin_ws/src/maps/src/markers.csv')
                    continue

            except KeyboardInterrupt:

                print("Bye")
                quit()

        cv2.destroyAllWindows()

if __name__=="__main__":

    ctx = getPoints('/proj/test_ws/src/simulate_result/data/map_level_6_b/l6_b_ed2.pgm')
    ctx.drive_gui()

