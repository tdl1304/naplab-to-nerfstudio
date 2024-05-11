from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from naplab.gps import GPSPoint
import matplotlib.pyplot as plt
from IPython.display import clear_output
import copy
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int, hidden_size: int, num_heads: int, dropout: float):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout),
            num_layers
        ).to(device)
        self.linear = nn.Linear(input_dim, hidden_size).to(device)
        self.decoder = nn.Linear(hidden_size, output_dim).to(device)
    
    def forward(self, src):
        src = self.linear(src.to(device))
        output = self.encoder(src)
        output = self.decoder(output)
        return output

class LinearInterpolationModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearInterpolationModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim).to(device)
    
    def forward(self, x):
        return self.linear(x.to(device))

class CoordinatePredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=3):
        super(CoordinatePredictionModel, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)

    def forward(self, x):
        return self.model(x)


def train_model(model, train_loader, optimizer, scheduler, num_epochs, pth_path="./models/best_model.pth"):
    best_rmse = float('inf')  # Initialize best RMSE
    losses = []
    criterion = nn.MSELoss()
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        total_rmse = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            rmse = torch.sqrt(loss)
            
            rmse.backward()  # Backward propagation using the direct loss
            
            optimizer.step()
            total_loss += loss.item()
            total_rmse += rmse.item()
            losses.append(rmse.item())
            
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        avg_rmse = total_rmse / len(train_loader)
        
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            torch.save(model.state_dict(), pth_path)  # Save the best model
        
        if (epoch+1) % 100 == 0:
            clear_output(wait=True)
            print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.8f}, Train RMSE: {avg_rmse:.8f}, Best RMSE: {best_rmse:.8f}, Optimizer LR: {scheduler.get_last_lr()}")
            plot_loss(losses, f"Training RMSE Loss (Epoch {epoch+1})")

def plot_loss(losses, title):
    plt.ion()
    plt.clf()
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.pause(0.001)  # Allow time for the plot to update

def predict(model, timestamp) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        output = model(timestamp.to(device)).cpu()
    return output

class GPSPointDataset(Dataset):
    def __init__(self, data: List[GPSPoint]):
        self.data = copy.deepcopy(data)
        test_timestamp = data[0].timestamp
        test_position = data[0].position
        
        timestamps = [gps_point.timestamp for gps_point in self.data]
        positions = np.stack([gps_point.position for gps_point in self.data])
        
        self.min_timestamp = min(timestamps)
        self.max_timestamp = max(timestamps)
        self.position_mean = positions.mean(axis=0)
        self.position_std = positions.std(axis=0)
        
        for gps_point in self.data:
            gps_point.timestamp = (gps_point.timestamp - self.min_timestamp) / (self.max_timestamp - self.min_timestamp)
            gps_point.position = (gps_point.position - self.position_mean) / self.position_std
        
        self.timestamp_tensors = torch.tensor(np.array([gps_point.timestamp for gps_point in self.data]), dtype=torch.float32).unsqueeze(1).to(device)
        self.position_tensors = torch.tensor(np.array([gps_point.position for gps_point in self.data]), dtype=torch.float32).to(device)
        
        if(self.denormalize_timestamp(self.data[0].timestamp) != test_timestamp):
            raise Exception("Error in denormalize_timestamp")
        if(self.denormalize_position(self.data[0].position).all() != test_position.all()):
            raise Exception("Error in denormalize_position")
    
    def __len__(self):
        return len(self.data)
    
    def normalize_1_timestamp(self, timestamp: int):
        return (timestamp - self.min_timestamp) / (self.max_timestamp - self.min_timestamp)
    
    def denormalize_timestamp(self, timestamp: float):
        return int(timestamp * (self.max_timestamp - self.min_timestamp) + self.min_timestamp)
    
    def denormalize_position(self, position: np.ndarray) -> np.ndarray:
        return (position * self.position_std + self.position_mean)
    
    def pop(self, idx):
        return self.data.pop(idx)
    
    def __getitem__(self, idx):
        return self.timestamp_tensors[idx], self.position_tensors[idx]