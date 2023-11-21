import os
import torch
import argparse
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch_geometric.nn import GraphUNet
from utils_simulated import *
from model import *

# ---------------------------
# seeding for reproducibility
# ---------------------------
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
# set CUBLAS_WORKSPACE_CONFIG=:16:8

def main():

    #-------------------------
    # arguments
    #-------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='model name')
    parser.add_argument('--simulation', required=True, help='model name')
    args = parser.parse_args()
    model = args.model
    simulation = args.simulation
    
    # ---------------------
    # parameters
    # ---------------------
    lr = 2e-3
    epochs = 200
    batch_size = 1
    pos_weights = 5
    path = 'D:graph-conflation-data/'
    
    # ---------------------
    # load data
    # ---------------------
    print('Load Datasets...')
    files = os.listdir(path+f'simulated-graphs/{simulation}/gt/')
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)
    
    # make datasets
    train_data = GraphDataset(path, simulation, train_files)
    val_data = GraphDataset(path, simulation, val_files)
    test_data = GraphDataset(path, simulation, test_files)
    
    # data loader
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)  
    
    # ---------------------
    #  models
    # ---------------------
    print('Load Model...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model == 'gcn':
        model_set1 = GCN().to(device)
        model_set2 = GCN().to(device)
    elif model == 'gat':
        model_set1 = GAT().to(device)
        model_set2 = GAT().to(device)
    elif model == 'graphsage':
        model_set1 = GraphSAGE().to(device)
        model_set2 = GraphSAGE().to(device)
    else:
        model_set1 = GraphUNet(2,64,128,3).to(device)
        model_set2 = GraphUNet(2,64,128,3).to(device)
    optimizer = torch.optim.Adam(list(model_set1.parameters()) + list(model_set2.parameters()), lr=lr)
    criterion= nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights))
    es = EarlyStopping(tolerance=10)
    
    # ---------------------
    # training
    # ---------------------    
    loss_track = []
    for epoch in range(epochs):
        
        # ----------------
        # Training
        # ----------------
        model_set1.train()
        model_set2.train()
        Train(
            train_dataloader, 
            model_set1, 
            model_set2, 
            optimizer, 
            criterion,
            device
        )
            
        # ----------------
        # Validation
        # ----------------
        model_set1.eval()
        model_set2.eval()
        val_loss = Eval(
            val_dataloader, 
            model_set1, 
            model_set2, 
            criterion,
            device
        )    
        loss_track.append(val_loss)
        
        # ----------------
        # Early Stop Check
        # ----------------
        es(val_loss)
        if es.early_stop:
            print(f" Early Stopping at Epoch {epoch}")
            print(f' Validation Loss: {round(val_loss, 5)}')
            break
        if es.save_model:     
            torch.save(model_set1.state_dict(), f'model_states/{model}_set1_{pos_weights}_{simulation}')
            torch.save(model_set2.state_dict(), f'model_states/{model}_set2_{pos_weights}_{simulation}')
            np.save(f'logs/{model}_simulation_losses_{pos_weights}_{simulation}.npy', loss_track)
        
        # ----------------
        # print val loss
        # ----------------
        print(f'Validation Loss: {val_loss}')
        
if __name__ == "__main__":
    main()