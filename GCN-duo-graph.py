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

def main():

    # ---------------------
    # parameters
    # ---------------------
    lr = 1e-2
    epochs = 200
    batch_size = 1
    pos_weights = 5
    
    # ---------------------
    # load data
    # ---------------------
    print('Load Datasets...')
    train_files = os.listdir('../graph-data/seattle-graphs/original/')
    test_files = os.listdir('../graph-data/west-seattle-graphs/original/')
    train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42)
    
    train_data = DuoGraphDataset('../graph-data/seattle-graphs/', train_files)
    val_data = DuoGraphDataset('../graph-data/seattle-graphs/', val_files)
    test_data = DuoGraphDataset('../graph-data/west-seattle-graphs/', test_files)
        
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)    
    
    # ---------------------
    #  models
    # ---------------------
    print('Load Model...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_del = GCN().to(device)
    model_ins = GCN().to(device)
    optimizer = torch.optim.Adam(list(model_del.parameters()) + list(model_ins.parameters()), lr=lr)
    criterion= nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights))
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.98)
    es = EarlyStopping()
    
    # ---------------------
    # training
    # ---------------------    
    loss_track = []
    for epoch in range(epochs):
        
        # ----------------
        # Training
        # ----------------
        model_del.train()
        model_ins.train()
        DuoTrain(
            train_dataloader, 
            model_del,
            model_ins,
            optimizer,
            criterion,
            device
        )
            
        # ----------------
        # Validation
        # ----------------
        model_del.eval()
        model_ins.eval()
        val_loss = DuoEvaluation(
            val_dataloader, 
            model_del,
            model_ins,
            optimizer,
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
            torch.save(model_del.state_dict(), f'model_states/DuoGraph_del_{lr}')
            torch.save(model_ins.state_dict(), f'model_states/DuoGraph_ins_{lr}')
            np.save(f'logs/DuoGraph_{lr}.npy', loss_track)
        
        # ----------------
        # print val loss
        # ----------------
        print(f'Validation Loss: {val_loss}')
        
if __name__ == "__main__":
    main()