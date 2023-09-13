### This code was created by Javier for ROS2 3.2 work package Jul 11 2023 ###

#!/usr/bin/env python3

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn_image as isns
import pandas as pd
import matplotlib_scalebar.scalebar as sb
import matplotlib.lines as mlines
#import pykrige.kriging_tools as kt
#from pykrige.uk import UniversalKriging
#from pykrige.ok import OrdinaryKriging
import scipy.stats as st
from   sklearn.datasets import make_blobs
import pandas as pd
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
import glob
#from pykrige.kriging_tools import write_asc_grid
#from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Path, PathPatch
#import gstools as gs
#import openturns as ot
import csv
import matplotlib.cm as cm
from scipy.interpolate import NearestNDInterpolator
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import normalize, MinMaxScaler

### Import Kriging ###

from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

######## Import files ########

import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename()

######## Determine boundary regions in the pgm map using circles ########

def isInside(circle_x, circle_y, rad, x, y):
     
    if ((x - circle_x) * (x - circle_x) +
        (y - circle_y) * (y - circle_y) <= rad * rad):
        return True;
    else:
        return False;

info 			= pd.read_csv('/home/javier/Desktop/human_tracking/mapl6_samples_100.csv')
info2 			= pd.read_csv('/home/javier/Desktop/human_tracking/mapl6_samples_100_2.csv')
mat  			= cv2.imread(image_path,0)
data 			= []

x_int_points 	= []
y_int_points 	= []
points 			= []

x_int_points    = info['x_points'].tolist()
y_int_points 	= info['y_points'].tolist()
den_vals    	= info['density'].tolist()

x_int_points2   = info2['x_points'].tolist()
y_int_points2	= info2['y_points'].tolist()
den_vals2    	= info2['density'].tolist()

us = []
vs = []

inf_point = []

need_inflation  = True
step_size 	= 12
sq_size 	= 40
inflation_rad = 1

if (len(x_int_points) != len(y_int_points)):
	print("Miss-match x and y co-ordinates error -1 !")
	quit()

# elif(len(x_int_points) != len(den_vals)):

# 	print("Miss-match x and y co-ordinates error -2 !")
# 	quit()	

else:
	for i in range(0, len(x_int_points)):
		points.append([x_int_points[i], y_int_points[i]])

	print("Loaded " + str(len(points)) + " interest points")
	# print(den_vals)

try:
	data = np.asarray( mat[:,:] )
	
except:
	print("Unable to read map data")
	quit()

if need_inflation:
	data_inflated = data.copy()
	for i in range(0, data_inflated.shape[0]): # For each row
		for j in range(0, data_inflated.shape[1]): # For each coloumn
			if data[j,i] == 0: # If colour is black in the pgm file, inflate it
				data_inflated = cv2.circle(data_inflated, (i,j), inflation_rad, 100, -1)				

######## Show boundary points that are inflated (ie the boundary that the algo has identified) ########

fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
implot = plt.imshow(data_inflated, cmap="bone")
plt.title('Base Map')
scalebar = sb.ScaleBar(dx=0.025, location='lower right', frameon='True') # 1 pixel = 0.05 meter
plt.gca().add_artist(scalebar)
plt.show()	
		
######## Show only areas within the boundary and make them white #######

data_mask 	= data_inflated.copy()
data_mask 	= np.where(data < 254, 0, data) # If value is less than 254 (ie not completely white) is True, return 0, else return data_mask 

fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
implot = plt.imshow(data_mask, cmap="bone")
plt.title('Masked Map')
plt.show()

######## Generate points within the boundary region #######

x_inside_points 	= []
y_inside_points 	= []
inside_points 		= []
#interest_points_x    = []
#interest_points_y    = []

for i in range(0, len(data_mask[0])): 
	for j in range(0, len(data_mask[1])):
		if data_mask[j,i] > 0: # If point is not black (ie the area that we want inside the boundary)
				
				inside_points.append([i,j])
				x_inside_points.append(i)
				y_inside_points.append(j)
#				interest_points_x.append(x_int_points[index])
#				interest_points_y.append(y_int_points[index])
#				simulated_density = np.random.randint(min(sim_range),max(sim_range), size=2)/100
#				dens_val.append(simulated_density[0])

#print(x_inside_points)
#print(y_inside_points)
#print(inside_points)
#print(type(inside_points))

################# Grid initiation (One for rbf, one for summation) #################

gridx = np.arange(0.0, 100, 1)
gridy = np.arange(0.0, 100, 1)
X, Y = np.mgrid[0:100:1, 0:100:1]

################# Estimation process for Radial Basis Function #################

rbf = Rbf(x_int_points, y_int_points, den_vals, function='linear') # Interpolate using the points and z values recorded
z_rbf = rbf(X, Y)

rbf2 = Rbf(x_int_points2, y_int_points2, den_vals2, function='linear') # Interpolate using the points and z values recorded
z_rbf2 = rbf2(X, Y)

### Only show z values for points inside boundary

rlu_values_rbf = []
 
for row in range(0,len(z_rbf)):
	for column in range(0,len(z_rbf[0])):
		if ([row,column] in inside_points):
			z_rbf[row][column] = z_rbf[row][column]
			#rlu_values_rbf.append(z_rbf[row][column])
		
		else:
			z_rbf[row][column] = 0

print(z_rbf[60][35])

for row in range(0,len(z_rbf2)):
	for column in range(0,len(z_rbf2[0])):
		if ([row,column] in inside_points):
			z_rbf[row][column] += z_rbf2[row][column]
			rlu_values_rbf.append(z_rbf[row][column])
		
		else:
			z_rbf[row][column] += 0

print(z_rbf2[60][35])
print(z_rbf[60][35])

print(np.max(rlu_values_rbf))
print(np.min(rlu_values_rbf))

############ Normalize 1d array ##############

rlu_values_rbf_min = np.min(rlu_values_rbf)
rlu_values_rbf_max = np.max(rlu_values_rbf)
rlu_values_rbf_range = rlu_values_rbf_max - rlu_values_rbf_min

rlu_normalized = []

for value in rlu_values_rbf:
	value = (value - rlu_values_rbf_min) / rlu_values_rbf_range 
	rlu_normalized.append(value)

print(np.max(rlu_normalized))
print(np.min(rlu_normalized))

############ Normalize 2d array ##############

for row in range(0,len(z_rbf)):
	for column in range(0,len(z_rbf[0])):
		if ([row,column] in inside_points):
			z_rbf[row][column] = (z_rbf[row][column] - rlu_values_rbf_min) / rlu_values_rbf_range
		
		else:
			z_rbf[row][column] = -1

z_rbf_transpose = z_rbf.transpose()

print(z_rbf_transpose[0][0])
print(np.amax(z_rbf_transpose))
print(np.amin(z_rbf_transpose))

### Plot figures
fig = plt.figure(figsize=(10,8))
ax = fig.gca()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.imshow(data, cmap="bone")
levels = np.linspace(min(rlu_normalized),max(rlu_normalized),21)
cfset = ax.contourf(gridx, gridy, z_rbf_transpose, levels = levels, cmap='RdYlGn_r') # vmin = min(rlu_values), vmax = max(rlu_values))
ax2 = fig.colorbar(cfset)
#plt.contour(gridx, gridy, zz, 3, colors='black')
#ax.scatter(x=x_int_points, y=y_int_points, s=200, c='green', cmap=cm.autumn_r, alpha=0.7, marker='.', edgecolor='none') # Sample dots
scalebar = sb.ScaleBar(dx=0.025, location='lower right', frameon='True', color = 'black') # 1 pixel = 0.05 meter
plt.gca().add_artist(scalebar)
#spot   = mlines.Line2D([], [], color='green', marker='.', linestyle='None', markersize=10, label='Sampled Point') # For the legend
#ax.legend(handles=[spot], loc='best')
ax2.set_label('Human activity score', fontsize = 15)
#plt.title('RLU value distribution map using Radial Basis Function')
#plt.grid('on')
plt.xticks(color='w')
plt.yticks(color='w')
plt.show()

























