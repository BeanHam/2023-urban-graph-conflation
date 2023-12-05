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
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

def main():

    #-------------------------
    # arguments
    #-------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='model name')
    parser.add_argument('--logits', required=True, help='model name')
    parser.add_argument('--pos_weight', required=True, help='model name')
    parser.add_argument('--simulation', required=True, help='model name')
    args = parser.parse_args()
    model = args.model
    logits = args.logits
    pos_weight = int(args.pos_weight)
    simulation = args.simulation
    
    # ---------------------
    # parameters
    # ---------------------
    lr = 2e-3
    epochs = 200
    batch_size = 1
    input_dim = 2
    hidden_dim = 32
    output_dim = 64
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
    graphconflator = GraphConflator(input_dim, hidden_dim, output_dim, model, logits).to(device)
    optimizer = torch.optim.Adam(graphconflator.parameters(), lr=lr)
    criterion= nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    es = EarlyStopping(tolerance=5)
    
    # ---------------------
    # training
    # ---------------------    
    loss_track = []
    for epoch in range(epochs):
        
        # ----------------
        # Training
        # ----------------
        graphconflator.train()
        Train(
            train_dataloader, 
            graphconflator,
            optimizer, 
            criterion,
            device
        )
            
        # ----------------
        # Validation
        # ----------------
        graphconflator.eval()
        val_loss = Eval(
            val_dataloader, 
            graphconflator,
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
            torch.save(graphconflator.state_dict(), f'model_states/graphconflator_{model}_{logits}_{pos_weight}_{simulation}')
            np.save(f'logs/graphconflator_{model}_{logits}_{pos_weight}_{simulation}.npy', loss_track)
        
        # ----------------
        # print val loss
        # ----------------
        print(f'Validation Loss: {val_loss}')
        
if __name__ == "__main__":
    main()