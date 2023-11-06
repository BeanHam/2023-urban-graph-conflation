import os
import torch
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import *
from model import *

seed=816
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main():

    # ---------------------
    # parameters
    # ---------------------
    lr = 1e-2
    epochs = 100
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
    model_osm = GCN().to(device)
    model_sdot = GCN().to(device)
    optimizer = torch.optim.Adam(list(model_osm.parameters()) + list(model_sdot.parameters()), lr=lr)
    criterion= nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights))
    es = EarlyStopping()
    
    # ---------------------
    # training
    # ---------------------    
    loss_track = []
    for e in range(epochs):
        
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
            criterion
        )
            
        # ----------------
        # Validation
        # ----------------
        model_del.eval()
        model_ins.eval()
        val_loss = Eval(
            val_dataloader, 
            model_osm, 
            model_sdot, 
            criterion
        )    
        loss_tract.append(val_loss)
        
        # ----------------
        # Early Stop Check
        # ----------------
        es(val_loss)
        if es.early_stop:
            print(f" Early Stopping at Epoch {epoch}")
            print(f' Validation Loss: {round(val_loss, 5)}')
            break
        if es.save_model:     
            torch.save(model_osm.state_dict(), f'model_states/model_osm_{pos_weights}')
            torch.save(model_sdot.state_dict(), f'model_states/model_sdot_{pos_weights}')
            np.save(f'logs/val_losses_{pos_weights}.npy', loss_track)
        
        # ----------------
        # print val loss
        # ----------------
        print(f'Validation Loss: {val_loss}')
        
if __name__ == "__main__":
    main()