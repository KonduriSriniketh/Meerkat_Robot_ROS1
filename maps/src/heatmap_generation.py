### This code was created by Javier for ROS2 3.2 work package Jul 11 2023 ###

#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import matplotlib_scalebar.scalebar as sb
from scipy.interpolate import Rbf
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Path, PathPatch
from PIL import Image
import yaml

######## RVIZ map coordinates to PGM map coordinates ########

# x axis is correct
# y axis need to flip ie (Height of pgm - y coordinate on rviz = y coordinate for pgm) 
# Remember to scale the raw pgm to rviz resolution dimension (multiply by eg 0.05) then to interpolation dimension (resulting value of the shortest width/height scale to 200)
# For x and y offset, a -50 x offset means that the position should be added by 50 before multiplying by 0.05

######## Filter the raw csv file to merge duplicate rows ######

# Assuming your CSV file is named 'data.csv', adjust the filename if necessary
file_path = '/home/sutd/catkin_ws/src/maps/src/human_positions.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Round each column (x, y, and count) to the nearest integer
df['x_points'] = df['x_points'].round().astype(int)
df['y_points'] = df['y_points'].round().astype(int)
# Group by x and y coordinates and sum the count column
df_merged = df.groupby(['x_points', 'y_points'], as_index=False)['density'].sum()

# Save the merged DataFrame back to a new CSV file
output_file_path = '/home/sutd/catkin_ws/src/maps/src/human_positions.csv'
df_merged.to_csv(output_file_path, index=False)

print("Rows with the same x and y coordinates have been merged and counts added. Result saved to 'merged_data.csv'")


######## Import pgm file ########

image_path = '/home/sutd/catkin_ws/src/maps/maps/level6.pgm'

######## Get yaml resolution and origin offset ########

def get_resolution_from_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        resolution = yaml_data['resolution']
        offset_x = yaml_data["origin"][0]
        offset_y = yaml_data["origin"][1]
        return resolution, offset_x, offset_y

yaml_file_path = '/home/sutd/catkin_ws/src/maps/maps/level6.yaml'
resolution, offset_x, offset_y = get_resolution_from_yaml(yaml_file_path)
print(f"Resolution: {resolution}")
print(offset_x, offset_y)

######## Transform RVIZ map coordinates to PGM map coordinates ########

info 			= pd.read_csv('/home/sutd/catkin_ws/src/maps/src/human_positions.csv')
mat  			= cv2.imread(image_path,0)
raw_height		= len(mat) # Height of the raw pgm file in pixels
raw_width		= len(mat[0]) # Width of the raw pgm file in pixels
rviz_height 	= round(len(mat) * resolution)
rviz_width 		= round(len(mat[0]) * resolution) 
print(rviz_width)
print(rviz_height)

data 			= []
points 			= []

x_original = info['x_points'].tolist()
y_original = info['y_points'].tolist()
den_vals    	= info['density'].tolist()
inf_point = []
print(x_original)
print(y_original)

need_inflation  = True
step_size 	= 12
sq_size 	= 40
inflation_rad = 1

######## Rescale the pgm file to the correct resolution (same as RVIZ) ########

def rescale_pgm(pgm_file_path, target_width, target_height):
    # Load the original .pgm image
    img = cv2.imread(pgm_file_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to the target dimensions
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    return resized_img

######## Rviz size ########
rviz_image = rescale_pgm(image_path, rviz_width, rviz_height)

######## Offset the vales and make them consistent from RVIZ to interpolation map ########
x_int_points = [element - offset_x for element in x_original] # x axis is aligned for both 
y_int_points = [(rviz_height - (element - offset_y)) for element in y_original] # y axis is flipped

######## Scale to interpolation size ########
if rviz_height > rviz_width:
	factor = 200 / rviz_width 
	height = round(rviz_height * factor)
	width = 200
	print("New height:", height)
	for value in range(0,len(x_int_points)):
		x_int_points[value] = round(x_int_points[value] * factor)
		y_int_points[value] = round(y_int_points[value] * factor) 	
	### Interpolation size ###
	rescaled_image = rescale_pgm(image_path, width, height)

elif rviz_width > rviz_height:
	factor = 200/ rviz_height 
	width = round(rviz_width * factor)
	height = 200 
	print("New width:", width)
	for value in range(0,len(x_int_points)):
		x_int_points[value] = round(x_int_points[value] * factor)
		y_int_points[value] = round(y_int_points[value] * factor) 
	### Interpolation size ###
	rescaled_image = rescale_pgm(image_path, width, height)

print(x_int_points)
print(y_int_points)

######## Inflate the identified boundary points ########

if (len(x_int_points) != len(y_int_points)):
	print("Miss-match x and y co-ordinates error -1 !")
	quit()

else:
	for i in range(0, len(x_int_points)):
		points.append([x_int_points[i], y_int_points[i]])

	print("Loaded " + str(len(points)) + " interest points")

try:
	data = np.asarray( mat[:,:] )
	
except:
	print("Unable to read map data")
	quit()

if need_inflation:
	data_inflated = rescaled_image.copy()
	for i in range(0, width): # For each row
		for j in range(0, height): # For each coloumn
			if rescaled_image[j,i] == 0: # If colour is black in the pgm file, inflate it
				data_inflated = cv2.circle(data_inflated, (i,j), inflation_rad, 100, -1)				

######## Show boundary points that are inflated (ie the boundary that the algo has identified) ########

print("I will start the base map")
fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_xlim(0, width)
ax.set_ylim(0, height)
implot = plt.imshow(data_inflated, cmap="bone")
plt.title('Base Map')
scalebar = sb.ScaleBar(dx=resolution, location='lower right', frameon='True') # 1 pixel = 0.05 meter
plt.gca().add_artist(scalebar)
plt.savefig('/home/sutd/catkin_ws/src/maps/src/human_activity/heatmap_spaces.png')
#plt.show()	 
		
######## Show only areas within the boundary and make them white #######

print("I will start the mask map")
data_mask 	= data_inflated.copy()
data_mask 	= np.where(rescaled_image < 254, 0, rescaled_image) # If value is less than 254 (ie not completely white) is True, return 0, else return data_mask 

fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_xlim(0, width)
ax.set_ylim(0, height)
implot = plt.imshow(data_mask, cmap="bone")
plt.title('Masked Map')
plt.savefig('/home/sutd/catkin_ws/src/maps/src//human_activity/heatmap_masked.png')
#plt.show() 

######## Generate points within the boundary region #######

print("I will start the identifying accessible points")
x_inside_points 	= []
y_inside_points 	= []
inside_points 		= []

for i in range(0, len(data_mask[0])): 
	for j in range(0, len(data_mask[1])):
		if data_mask[j,i] > 0: # If point is not black (ie the area that we want inside the boundary)
				
				inside_points.append([i,j])
				x_inside_points.append(i)
				y_inside_points.append(j)

######## Grid initiation (One for rbf, one for summation) ########

gridx = np.arange(0.0, width, 1)
gridy = np.arange(0.0, height, 1)
X, Y = np.mgrid[0:width:1, 0:height:1]

######## Estimation process for Radial Basis Function ########

rbf = Rbf(x_int_points, y_int_points, den_vals, function='linear') # Interpolate using the points and z values recorded
z_rbf = rbf(X, Y)

### Only show z values for points inside boundary

rlu_values_rbf = []
 
for row in range(0,len(z_rbf)):
	for column in range(0,len(z_rbf[0])):
		if ([row,column] in inside_points):
			z_rbf[row][column] = z_rbf[row][column]
			rlu_values_rbf.append(z_rbf[row][column])
		
		else:
			z_rbf[row][column] = 0

print(np.max(rlu_values_rbf))
print(np.min(rlu_values_rbf))

######## Normalize 1d array ########

rlu_values_rbf_min = np.min(rlu_values_rbf)
rlu_values_rbf_max = np.max(rlu_values_rbf)
rlu_values_rbf_range = rlu_values_rbf_max - rlu_values_rbf_min

rlu_normalized = []

for value in rlu_values_rbf:
	value = (value - rlu_values_rbf_min) / rlu_values_rbf_range 
	rlu_normalized.append(value)

print(np.max(rlu_normalized))
print(np.min(rlu_normalized))

######## Normalize 2d array ########

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

######## Plot figures ########

print("I will start plotting figures")
fig = plt.figure(figsize=(10,8))
ax = fig.gca()
ax.set_xlim(0, width)
ax.set_ylim(0, height)
print("Figure set-up done")

levels = np.linspace(min(rlu_normalized),max(rlu_normalized),21)
cfset = ax.contourf(gridx, gridy, z_rbf_transpose, levels = levels, cmap='RdYlGn_r') # vmin = min(rlu_values), vmax = max(rlu_values))
ax2 = fig.colorbar(cfset)
ax2.set_label('Human activity score', fontsize = 15)
print("Colorbar set-up done")

scalebar = sb.ScaleBar(dx=resolution, location='lower right', frameon='True', color = 'black') # 1 pixel = 0.05 meter
plt.gca().add_artist(scalebar)
plt.title('Human activity heatmap from 8am to 11am', fontsize = 15)
plt.xticks(color='w')
plt.yticks(color='w')
#plt.show()
plt.savefig('/home/sutd/catkin_ws/src/maps/src/human_activity/heatmap.png')





















