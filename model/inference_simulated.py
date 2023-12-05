import os
import torch
import argparse
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch_geometric.nn import GraphUNet
from utils_simulated import *
from model import *
import warnings
warnings.filterwarnings('ignore')

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
    print('===============================')
    print(f'Model: {model}')
    print(f'Logits: {logits}')
    print(f'POS Weight: {pos_weight}')
    print(f'Simulation: {simulation}')    
    
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
    graphconflator.load_state_dict(torch.load(f'model_states/graphconflator_{model}_{logits}_{pos_weight}_{simulation}'))
    optimizer = torch.optim.Adam(graphconflator.parameters(), lr=lr)
    criterion= nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    es = EarlyStopping(tolerance=10)
    
    # ---------------------
    #  graph level accuracy
    # ---------------------    
    graph_acc = []
    graph_fp = []
    graph_fn = []
    for i, batch in enumerate(tqdm(test_dataloader)):
        
        if i == 1591: continue
        
        # load data
        graph_gt, graph_set1, graph_set2, gt_x, x_set1, x_set2 = batch
        graph_gt = graph_gt.squeeze_(0).to(device)
        labels = graph_gt.flatten()
        graph_set1 = graph_set1.squeeze_(0).long().to(device)
        graph_set2 = graph_set2.squeeze_(0).long().to(device)
        gt_x = gt_x.squeeze_(0).to(device)
        x_set1 = x_set1.squeeze_(0).to(device)
        x_set2 = x_set2.squeeze_(0).to(device)
        
        # make prediction
        with torch.no_grad():
            logits = graphconflator(graph_set1, graph_set2, x_set1, x_set2)  
        A = torch.sigmoid(logits).flatten()
        tn, fp, fn, tp = confusion_matrix(1.0*(A.detach()>0.5).cpu(), labels.cpu(), labels=[1,0]).ravel()
        graph_acc.append((tn+tp)/len(A))
        graph_fp.append(fp/(fp+tn))
        graph_fn.append(fn/(tp+fn))
    
    # ---------------------
    #  edge level accuracy
    # ---------------------    
    edge_acc = []
    edge_fp = []
    edge_fn = []
    for i, batch in enumerate(tqdm(test_dataloader)):
        
        if i == 1591: continue
            
        # load data
        graph_gt, graph_set1, graph_set2, gt_x, x_set1, x_set2 = batch
        graph_gt = graph_gt.squeeze_(0).to(device)
        labels = graph_gt.flatten()
        graph_set1 = graph_set1.squeeze_(0).long().to(device)
        graph_set2 = graph_set2.squeeze_(0).long().to(device)
        gt_x = gt_x.squeeze_(0).to(device)
        x_set1 = x_set1.squeeze_(0).to(device)
        x_set2 = x_set2.squeeze_(0).to(device)
        
        # extract conflicted edges
        conflicted_edges = torch.concatenate(
            [graph_set1.T[~(graph_set1.T[:, None] == graph_set2.T).all(-1).any(-1)],
             graph_set2.T[~(graph_set2.T[:, None] == graph_set1.T).all(-1).any(-1)]],
            axis=0
        ).cpu().detach().numpy()
        conflicted_edges = np.repeat(conflicted_edges, 2,axis=0)
        conflicted_edges[::2,[0,1]] = conflicted_edges[::2,[1,0]]
        conflicted_edges = list(zip(*conflicted_edges.T))     
        
        # make prediction
        with torch.no_grad():
            logits = graphconflator(graph_set1, graph_set2, x_set1, x_set2)  
        A = torch.sigmoid(logits)
        A_pred = 1.0*(A.detach().cpu()>0.5)
        
        preds = [A_pred[c].item() for c in conflicted_edges]
        gts = [graph_gt[c].item() for c in conflicted_edges]
        tn, fp, fn, tp = confusion_matrix(preds, gts, labels=[1,0]).ravel()
        edge_acc.append((tn+tp)/len(A))
        edge_fp.append(fp/(fp+tn))
        edge_fn.append(fn/(tp+fn))
    
    print('Graph Level: ')
    print('----------------------------------------')
    print(f'Accuracy: {np.mean(graph_acc)}')
    print(f'FP Rate: {np.mean(np.array(graph_fp)[~np.isnan(graph_fp)])}')
    print(f'FN Rate: {np.mean(np.array(graph_fn)[~np.isnan(graph_fn)])}')
    
    print('Edge Level: ')
    print('----------------------------------------')
    print(f'Accuracy: {np.mean(edge_acc)}')
    print(f'FP Rate: {np.mean(np.array(edge_fp)[~np.isnan(edge_fp)])}')
    print(f'FN Rate: {np.mean(np.array(edge_fn)[~np.isnan(edge_fn)])}')
    
if __name__ == "__main__":
    main()