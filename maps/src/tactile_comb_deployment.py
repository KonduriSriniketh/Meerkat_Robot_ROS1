### This code was created by Javier for ROS2 3.2 work package Jul 11 2023 ###

import pandas as pd
import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from gnn_class import DATA
from scipy.stats import mode
import serial
import time

serial_port = '/dev/ttyACM0'  # Replace with the actual port name
baud_rate = 9600

ser = serial.Serial(serial_port, baud_rate)

switch_states = []  # Empty list to store switch states
count = 0
data = []

print('Priming')

time.sleep(2)

print('Lets go!')

time.sleep(2)

while True:
    try:
        time.sleep(0.5)
        line = ser.readline().decode('latin-1').strip()
        if line:  # Check if the line is not empty
            switch_values = line.split(" ")
            switch_states = [int(value) for value in switch_values]
            if len(switch_states) == 10:
                data.append(switch_states)
                print(switch_states)
                count += 1
        
        print(count)
        if count == 10:
            break

    except KeyboardInterrupt:
        break

ser.close()
print(data)

start = time.process_time()

""" data = [[0, 1 , 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1 , 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1 , 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1 , 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1 , 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1 , 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1 , 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1 , 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1 , 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1 , 0, 1, 0, 1, 0, 1, 0, 1]] """

""" data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] """

state_dict = torch.load('/home/sutd/meerkat_ros_ws/src/maps/weights/best_model.pt', map_location='cpu')

### Pre-process data

df = pd.DataFrame(columns=[['pos'+ str(i) for i in range(0,100)] + ['weight']])

new_data_list = []
new_data_list_edge = []
new_data_list_weight = []

for row in range(0, len(data)):
    for column in range(0, len(data)):
        new_data_list.append(data[row][column])

for index in range(0, len(data)):

        column_array = []
        count = 0
        
        for index2 in range(0, len(data)):
            column_array.append(new_data_list[count])
            count += 10

        changes = (np.diff(column_array)!=0).sum()
        new_data_list_edge.append(int(changes))

### Calculate the number of zeroes per row ### 

zero_counts = new_data_list.count(0)
total_counts = sum(new_data_list_edge) + zero_counts
new_data_list_weight.append(int((zero_counts + total_counts) / 2))
new_data = new_data_list + new_data_list_weight
df.loc[len(df)] = new_data 
df['label'] = 8

# append the same row to the DataFrame 10 times
new_rows = [df] * 10
df = pd.concat(new_rows, ignore_index=True)

# add an index column in ascending order with header "node_id"
df = df.reset_index(drop=False).rename(columns={'index': 'node_id'})

processed_data = DATA(df).main()

print(processed_data.validate(raise_on_error=True))
print(processed_data)
print(type(processed_data))
print(type(processed_data.x))
print(type(processed_data.y))
print(type(processed_data.edge_index))
print(len(processed_data.test_mask) == processed_data.num_nodes)
print(processed_data.test_mask)
print(processed_data.num_features)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {processed_data.num_nodes}')
print(f'Number of edges: {processed_data.num_edges}')
print(f'Average node degree: {processed_data.num_edges / processed_data.num_nodes:.2f}')
print(f'Number of unique classes: {processed_data.num_classes}')

class CHEB(torch.nn.Module):
    def __init__(self, out_channels):
        super(CHEB, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.conv1 = ChebConv(processed_data.num_features, out_channels, K =1)
        self.conv2 = ChebConv(out_channels , out_channels, K =1)
        self.out = Linear(out_channels , 8)

    def forward(self, x, edge_index, edge_weight):
        
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Output layer 
        x = F.softmax(self.out(x), dim=1)
        return x

gcn_model = CHEB(out_channels=64)
print(gcn_model)

### Load saved weights into model ###
gcn_model.load_state_dict(state_dict)
gcn_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gcn_model = gcn_model.to(gcn_device)
processed_data = processed_data.to(gcn_device)

### Setup params ###
gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr = 0.007598396573086836, weight_decay= 0.0000612817825780508)
gcn_criterion = torch.nn.CrossEntropyLoss() 

### Test function ###
gcn_model.eval()

with torch.no_grad():

    test_out = gcn_model(processed_data.x, processed_data.edge_index, processed_data.edge_weight)

    pred = test_out.argmax(dim=1)  # Use the class with highest probability.
    final_pred_mode = mode(pred.cpu())
    final_pred = int(final_pred_mode.mode) + 1
    print(final_pred)
    print(f"Predicted PATH tile class: ", final_pred)

print(time.process_time() - start)
    
