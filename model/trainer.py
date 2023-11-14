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
from utils import *
from model import *

seed=816
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main():

    #-------------------------
    # arguments
    #-------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='model name')
    args = parser.parse_args()
    model = args.model
    
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
    files = os.listdir(path+'/graphs/osm/')
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)
    
    # make datasets
    train_data = GraphDataset(path, train_files)
    val_data = GraphDataset(path, val_files)
    test_data = GraphDataset(path, test_files)
    
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
        model_osm = GCN().to(device)
        model_sdot = GCN().to(device)
    elif model == 'gat':
        model_osm = GAT().to(device)
        model_sdot = GAT().to(device)
    elif model == 'graphsage':
        model_osm = GraphSAGE().to(device)
        model_sdot = GraphSAGE().to(device)
    else:
        model_osm = GraphUNet(2,64,128,3).to(device)
        model_sdot = GraphUNet(2,64,128,3).to(device)
    optimizer = torch.optim.Adam(list(model_osm.parameters()) + list(model_sdot.parameters()), lr=lr)
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
        model_osm.train()
        model_sdot.train()
        Train(
            train_dataloader, 
            model_osm, 
            model_sdot, 
            optimizer, 
            criterion,
            device
        )
            
        # ----------------
        # Validation
        # ----------------
        model_osm.eval()
        model_sdot.eval()
        val_loss = Eval(
            val_dataloader, 
            model_osm, 
            model_sdot, 
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
            torch.save(model_osm.state_dict(), f'model_states/{model}_osm_{pos_weights}')
            torch.save(model_sdot.state_dict(), f'model_states/{model}_sdot_{pos_weights}')
            np.save(f'logs/{model}_losses_{pos_weights}.npy', loss_track)
        
        # ----------------
        # print val loss
        # ----------------
        print(f'Validation Loss: {val_loss}')
        
if __name__ == "__main__":
    main()