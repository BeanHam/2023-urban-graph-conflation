import os
import json
import torch
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
from tqdm import tqdm

seed=816
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ---------------------
# Load Attributes
# ---------------------
def load_attributes(path, min_bounds, max_bounds):
    with open(path) as f: x = json.load(f)        
    x = torch.from_numpy(np.stack([v for k,v in x.items()])).float()    
    x = (x-min_bounds)/(max_bounds-min_bounds)
    return x

# ---------------------
# Data Loader
# ---------------------
class GraphDataset(Dataset):
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
        min_bounds = torch.tensor([-122.436939, 47.495553])
        max_bounds = torch.tensor([-122.236231, 47.734147])        
        
        # paths
        osw_path = os.path.join(self.graph_dir, 'graphs/osw/', file)
        osm_path = os.path.join(self.graph_dir, 'graphs/osm/', file)
        sdot_path = os.path.join(self.graph_dir, 'graphs/sdot/', file)        
        osw_x_path = os.path.join(self.graph_dir, 'attributes/osw/', file+'.json')
        osm_x_path = os.path.join(self.graph_dir, 'attributes/osm/', file+'.json')
        sdot_x_path = os.path.join(self.graph_dir, 'attributes/sdot/', file+'.json')
        
        # graphs & edges
        graph_osw = torch.from_numpy(nx.to_numpy_array(nx.read_graph6(osw_path))).float()
        graph_osm = nx.read_graph6(osm_path)
        graph_sdot = nx.read_graph6(sdot_path)        
        graph_osm = np.array(graph_osm.edges()).T
        graph_sdot = np.array(graph_sdot.edges()).T
        
        # graph attributes
        osw_x = load_attributes(osw_x_path, min_bounds, max_bounds)
        osm_x = load_attributes(osm_x_path, min_bounds, max_bounds)
        sdot_x = load_attributes(sdot_x_path, min_bounds, max_bounds)

        return graph_osw, graph_osm, graph_sdot, osw_x, osm_x, sdot_x

# ---------------------
# Trainer
# ---------------------
def Train(train_data, 
          model_osm, 
          model_sdot, 
          optimizer, 
          criterion, 
          device):
    
    for i, batch in enumerate(tqdm(train_data)):
        
        # load data
        graph_osw, graph_osm, graph_sdot, osw_x, osm_x, sdot_x = batch
        graph_osw = graph_osw.squeeze_(0).to(device)
        labels = graph_osw.flatten()
        graph_osm = graph_osm.squeeze_(0).long().to(device)
        graph_sdot = graph_sdot.squeeze_(0).long().to(device)
        osw_x = osw_x.squeeze_(0).to(device)
        osm_x = osm_x.squeeze_(0).to(device)
        sdot_x = sdot_x.squeeze_(0).to(device)
        
        # make prediction
        optimizer.zero_grad() 
        out_osm = model_osm(osm_x, graph_osm)
        out_sdot = model_sdot(sdot_x, graph_sdot)
        logits = torch.matmul(out_osm, out_sdot.T).flatten()
        loss = criterion(logits, labels)
        loss.backward() 
        optimizer.step()
        
# ---------------------
# Evaluation
# ---------------------
def Eval(val_data, 
         model_osm, 
         model_sdot, 
         criterion, 
         device):

    val_losses = []
    for i, batch in enumerate(val_data):
        
        # load data
        graph_osw, graph_osm, graph_sdot, osw_x, osm_x, sdot_x = batch
        graph_osw = graph_osw.squeeze_(0).to(device)
        labels = graph_osw.flatten()
        graph_osm = graph_osm.squeeze_(0).long().to(device)
        graph_sdot = graph_sdot.squeeze_(0).long().to(device)    
        osw_x = osw_x.squeeze_(0).to(device)
        osm_x = osm_x.squeeze_(0).to(device)
        sdot_x = sdot_x.squeeze_(0).to(device)
         
        with torch.no_grad():
            out_osm = model_osm(osm_x, graph_osm)
            out_sdot = model_sdot(sdot_x, graph_sdot)
        logits = torch.matmul(out_osm, out_sdot.T).flatten()
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