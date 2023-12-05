import os
import json
import torch
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
from tqdm import tqdm

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
                 simulation,
                 graph_files):
        
        self.graph_dir = graph_dir
        self.simulation = simulation
        self.graph_files = graph_files     
    
    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        
        # single graph
        file = self.graph_files[idx]
        min_bounds = torch.tensor([-122.436939, 47.495553])
        max_bounds = torch.tensor([-122.236231, 47.734147])        
        
        # paths
        gt_path = os.path.join(self.graph_dir, f'simulated-graphs/{self.simulation}/gt/', file)
        set1_path = os.path.join(self.graph_dir, f'simulated-graphs/{self.simulation}/set1/', file)
        set2_path = os.path.join(self.graph_dir, f'simulated-graphs/{self.simulation}/set2/', file)        
        gt_x_path = os.path.join(self.graph_dir, f'simulated-attributes/{self.simulation}/gt/', file+'.json')
        x_set1_path = os.path.join(self.graph_dir, f'simulated-attributes/{self.simulation}/set1/', file+'.json')
        x_set2_path = os.path.join(self.graph_dir, f'simulated-attributes/{self.simulation}/set2/', file+'.json')
        
        # graphs & edges
        graph_gt = torch.from_numpy(nx.to_numpy_array(nx.read_graph6(gt_path))).float()
        graph_set1 = nx.read_graph6(set1_path)
        graph_set2 = nx.read_graph6(set2_path)        
        graph_set1 = np.array(graph_set1.edges()).T
        graph_set2 = np.array(graph_set2.edges()).T
        
        # graph attributes
        gt_x = load_attributes(gt_x_path, min_bounds, max_bounds)
        x_set1 = load_attributes(x_set1_path, min_bounds, max_bounds)
        x_set2 = load_attributes(x_set2_path, min_bounds, max_bounds)

        return graph_gt, graph_set1, graph_set2, gt_x, x_set1, x_set2

# ---------------------
# Trainer
# ---------------------
def Train(train_data, 
          graphconflator,
          optimizer, 
          criterion, 
          device):
    
    for i, batch in enumerate(tqdm(train_data)):
        
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
        optimizer.zero_grad() 
        logits = graphconflator(graph_set1, graph_set2, x_set1, x_set2)        
        loss = criterion(logits.flatten(), labels)
        loss.backward() 
        optimizer.step()
        
# ---------------------
# Evaluation
# ---------------------
def Eval(val_data, 
         graphconflator, 
         criterion, 
         device):

    val_losses = []
    for i, batch in enumerate(val_data):
        
        # load data
        graph_gt, graph_set1, graph_set2, gt_x, x_set1, x_set2 = batch
        graph_gt = graph_gt.squeeze_(0).to(device)
        labels = graph_gt.flatten()
        graph_set1 = graph_set1.squeeze_(0).long().to(device)
        graph_set2 = graph_set2.squeeze_(0).long().to(device)
        gt_x = gt_x.squeeze_(0).to(device)
        x_set1 = x_set1.squeeze_(0).to(device)
        x_set2 = x_set2.squeeze_(0).to(device)
         
        with torch.no_grad():
            logits = graphconflator(graph_set1, graph_set2, x_set1, x_set2)
        loss = criterion(logits.flatten(), labels)
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