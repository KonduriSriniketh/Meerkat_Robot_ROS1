### This work was done by Javier for GNN Paper and ROS2 3.2 work package Jul 11 2023 ###

import numpy as np
import torch
import itertools
from torch_geometric.data import Data

class DATA:
    def __init__(self, data):
        self.data = data
    
    def main(self):
        ## Edge index using exact same ###

        self.data['combined'] = self.data.apply(lambda row: int(''.join(map(str, row[['pos'+ str(i) for i in range(0,100)]]))), axis=1)

        team = [self.data.iloc[0]["combined"]]
        self.all_edges = np.array([], dtype=np.int32).reshape((0, 2))

        team_df = self.data
        players = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # Build all combinations, as all players are connected
        permutations = list(itertools.combinations(players, 2))
        edges_source = [e[0] for e in permutations]
        edges_target = [e[1] for e in permutations]
        team_edges = np.column_stack([edges_source, edges_target])
        self.all_edges = np.vstack([self.all_edges, team_edges])

        # Convert to Pytorch Geometric format
        edge_index = torch.tensor(self.all_edges.transpose(), dtype=torch.int64)

        # Get node features
        node_feats = self.get_node_features(self.data)

        # Get node labels
        node_labels = self.get_node_labels(self.data)

        # Get number of classes
        num_classes = self.get_num_classes(self.data)

        # Get edge weights 
        edge_weight = self.get_edge_weight(self.data, self.all_edges)

        compiled_data = Data(x=node_feats, edge_index=edge_index, edge_weight=edge_weight, y=node_labels, num_classes=num_classes)
        
        # Get masks (Training, Validation, Test)
        test_mask = self.get_masks(self.data, compiled_data)
        print(test_mask)

        compiled_data.test_mask = test_mask

        print(compiled_data.validate(raise_on_error=True))
        print(compiled_data)
        print(type(compiled_data))
        print(type(compiled_data.x))
        print(type(compiled_data.y))
        print(type(compiled_data.edge_index))
        print(len(compiled_data.test_mask) == compiled_data.num_nodes)
        print(compiled_data.test_mask)
        print(compiled_data.num_features)
        print('===========================================================================================================')

        # Gather some statistics about the graph.
        print(f'Number of nodes: {compiled_data.num_nodes}')
        print(f'Number of edges: {compiled_data.num_edges}')
        print(f'Average node degree: {compiled_data.num_edges / compiled_data.num_nodes:.2f}')
        print(f'Number of unique classes: {compiled_data.num_classes}')

        return compiled_data

    def get_node_features(self, data):

        node_features = data[['pos'+ str(i) for i in range(0,100)]]
        node_features.head()
        x = torch.tensor(node_features.to_numpy(), dtype=torch.float)

        return x

    def get_node_labels(self, data):

        labels = data[["label"]]
        labels.head()
        y1 = (labels.to_numpy())
        y = torch.tensor(y1.ravel(), dtype=torch.int64)
        
        return y

    def get_num_classes(self, data):

        num_classes = 1
        
        return num_classes
        
    def get_edge_weight(self, data, all_edges):

        all_edge_weights = []
        edge_index_int_weight = 0

        for edge in all_edges:
            
            edge_weights = int(data.iloc[0]["weight"])

            all_edge_weights.append(round(edge_weights, 2))

            edge_index_int_weight += 1
                
        all_edge_weights = np.asarray(all_edge_weights)
        print(np.shape(all_edge_weights))
        edge_weight = torch.tensor(all_edge_weights, dtype=torch.float32)

        return edge_weight

    def get_masks(self, data, compiled_data):

        test_indices = []

        # Define the class labels
        classes = data["label"]

        for label in classes:
            # Filter the data to only include the current class
            class_data = data[data["label"] == label]

            # Split the indices of the class into training, validation, and test sets
            n = len(class_data)
            test_indices.extend(class_data.index[:])

        print(test_indices)

        test_mask = torch.zeros(compiled_data.num_nodes, dtype=torch.bool)
        test_mask[test_indices] = 1

        return test_mask
