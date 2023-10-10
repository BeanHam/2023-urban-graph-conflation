import os
import torch
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
from tqdm import tqdm

# ---------------------
# DuoGraph Data Loader
# ---------------------
class DuoGraphDataset(Dataset):
    def __init__(self, 
                 graph_dir, 
                 graph_files):
        
        self.graph_dir = graph_dir
        self.graph_files = graph_files        
    
    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        
        # single graph
        file = self.graph_files[idx]
        
        # graph paths
        graph_gt_path = os.path.join(self.graph_dir, 'original', file)
        graph_del_path = os.path.join(self.graph_dir, 'deletion', file)
        graph_ins_path = os.path.join(self.graph_dir, 'insertion', file)
        
        # ground truth adj
        graph_gt = torch.from_numpy(nx.to_numpy_array(nx.read_gpickle(graph_gt_path))).float()
        
        # deletion graph embedding
        graph_del = nx.read_gpickle(graph_del_path)
        graph_ins = nx.read_gpickle(graph_ins_path)        
        graph_del_edge_index = np.array(graph_del.edges()).T
        graph_ins_edge_index = np.array(graph_ins.edges()).T
        x = torch.ones(len(graph_del),1)
        
        return graph_gt, graph_del_edge_index, graph_ins_edge_index, x

# ---------------------
# Single Graph Data Loader
# ---------------------    
class SingleGraphDataset(Dataset):
    def __init__(self, 
                 graph_dir, 
                 graph_files):
        
        self.graph_dir = graph_dir
        self.graph_files = graph_files        
    
    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        
        # single graph
        file = self.graph_files[idx]
        
        # graph paths
        graph_gt_path = os.path.join(self.graph_dir, 'original', file)
        graph_del_path = os.path.join(self.graph_dir, 'deletion', file)
        graph_ins_path = os.path.join(self.graph_dir, 'insertion', file)
        
        # ground truth adj
        graph_gt = torch.from_numpy(nx.to_numpy_array(nx.read_gpickle(graph_gt_path))).float()
        
        # deletion graph embedding
        graph_del = nx.read_gpickle(graph_del_path)
        graph_ins = nx.read_gpickle(graph_ins_path)
        graph_comb = nx.compose(graph_ins, graph_del)
        
        edge_index = np.array(graph_comb.edges()).T
        x = torch.ones(len(graph_comb),1)
        
        return graph_gt, edge_index, x    
    
# ---------------------
# DuoGraph Training Function
# ---------------------
def DuoTrain(dataloader,
             model_del,
             model_ins,
             optimizer,
             criterion,
             device):
    
    for i, batch in enumerate(tqdm(dataloader)):
            
        # load data
        graph_gt, graph_del_edge_index, graph_ins_edge_index, x = batch
        graph_gt = graph_gt.squeeze_(0).to(device)
        labels = graph_gt.flatten()
        graph_del_edge_index = graph_del_edge_index.squeeze_(0).to(device)
        graph_ins_edge_index = graph_ins_edge_index.squeeze_(0).to(device)    
        x = x.squeeze_(0).to(device)
        
        # make prediction
        optimizer.zero_grad() 
        out_del = model_del(graph_del_edge_index, x)
        out_ins = model_ins(graph_ins_edge_index, x) 
        logits = torch.matmul(out_del, out_ins.T).flatten()
        loss = criterion(logits, labels)
        loss.backward() 
        optimizer.step()

        
# ---------------------
# DuoGraph Training Function
# ---------------------
def SingleTrain(dataloader,
                model,
                optimizer,
                criterion,
                device):
    
    for i, batch in enumerate(tqdm(dataloader)):
            
        # load data
        graph_gt, edge_index, x = batch
        graph_gt = graph_gt.squeeze_(0).to(device)
        labels = graph_gt.flatten()
        edge_index = edge_index.squeeze_(0).to(device)
        x = x.squeeze_(0).to(device)
        
        # make prediction
        optimizer.zero_grad()
        out = model(edge_index, x)
        logits = torch.matmul(out, out.T).flatten()
        loss = criterion(logits, labels)
        loss.backward() 
        optimizer.step()
        
# ---------------------
# Training Function
# ---------------------
def DuoEvaluation(dataloader,
               model_del,
               model_ins,
               optimizer,
               criterion,
               device):
    
    val_losses = []
    for i, batch in enumerate(dataloader):
        
        # load data
        graph_gt, graph_del_edge_index, graph_ins_edge_index, x = batch
        graph_gt = graph_gt.squeeze_(0).to(device)
        labels = graph_gt.flatten()
        graph_del_edge_index = graph_del_edge_index.squeeze_(0).to(device)
        graph_ins_edge_index = graph_ins_edge_index.squeeze_(0).to(device)    
        x = x.squeeze_(0).to(device)
        
        # make prediction
        with torch.no_grad():
            out_del = model_del(graph_del_edge_index, x)
            out_ins = model_ins(graph_ins_edge_index, x) 
        logits = torch.matmul(out_del, out_ins.T).flatten()
        loss = criterion(logits, labels)
        val_losses.append(loss.item())
     
    return np.mean(val_losses)

# ---------------------
# Training Function
# ---------------------
def SingleEvaluation(dataloader,
                     model,
                     optimizer,
                     criterion,
                     device):
    
    val_losses = []
    for i, batch in enumerate(dataloader):
        
        # load data
        graph_gt, edge_index, x = batch
        graph_gt = graph_gt.squeeze_(0).to(device)
        labels = graph_gt.flatten()
        edge_index = edge_index.squeeze_(0).to(device)
        x = x.squeeze_(0).to(device)
        
        # make prediction
        with torch.no_grad():
            out = model(edge_index, x)
        logits = torch.matmul(out, out.T).flatten()      
        loss = criterion(logits, labels)
        val_losses.append(loss.item())
     
    return np.mean(val_losses)

# ---------------------
# Early Stop Function
# ---------------------
class EarlyStopping():
    def __init__(self, 
                 tolerance=5):

        self.tolerance = tolerance
        self.loss_min = np.inf
        self.counter = 0
        self.early_stop = False
        self.save_model = False
        
    def __call__(self, loss):
        if loss > self.loss_min:
            self.counter +=1
            self.save_model = False
            if self.counter >= self.tolerance:  
                self.early_stop = True
        else:
            self.save_model = True
            self.loss_min = loss
            self.counter = 0    