import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import glob
import itertools
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

from core.model import OrthogonalGDIFNet
from core.graph_constructor import create_2d_grid_graph

class XaiRiskDataset(Dataset):
    def __init__(self, root):
        self.filenames = sorted(glob.glob(os.path.join(root, '*.npz')))
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        with np.load(self.filenames[idx]) as file:
            data = torch.from_numpy(file['data'].astype(np.float32))
            label = torch.from_numpy(file['label'].astype(np.float32))
        return data, label

def orthogonality_loss(latent_risks_dict):
    loss = 0.0; count = 0
    for (risk1, t1), (risk2, t2) in itertools.combinations(latent_risks_dict.items(), 2):
        cos_sim = F.cosine_similarity(t1, t2, dim=1).mean()
        loss += torch.abs(cos_sim)
        count += 1
    return loss / count if count > 0 else torch.tensor(0.0)

def train_all_scenarios():
    with open('configs/default_config.json', 'r') as f: config = json.load(f)
    with open('configs/scenario_weights.json', 'r') as f: scenario_configs = json.load(f)
    
    DIMS, RISK_TYPES = config['grid']['dimensions'], config['risk_types']
    MODEL_CFG, TRAIN_CFG = config['model'], config['training']
    DATA_DIR, MODELS_DIR = "training_data_xai", "models"
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)

    input_map = {
        "collision": ["static_obstacles", "no_fly_zones"], "exposure": ["radar_coverage", "camera_coverage"],
        "energy": ["uav_state_battery"], "communication": ["comm_signal"], "threat": ["hostile_threat"]
    }
    data_channels = [
        "static_obstacles", "no_fly_zones", "radar_coverage", "camera_coverage",
        "comm_signal", "hostile_threat", "uav_state_battery"
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = create_2d_grid_graph(DIMS).to(device)

    for scenario_name in scenario_configs.keys():
        print(f"\n{'='*20} 正在训练 '{scenario_name}' 场景模型 {'='*20}")
        
        scenario_data_dir = os.path.join(DATA_DIR, scenario_name)
        if not os.path.exists(scenario_data_dir) or not os.listdir(scenario_data_dir):
            print(f"警告: 未找到 '{scenario_name}' 的数据，跳过训练。请先运行 generate_dataset.py")
            continue
            
        full_dataset = XaiRiskDataset(scenario_data_dir)
        val_size = int(TRAIN_CFG['validation_split'] * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=TRAIN_CFG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=TRAIN_CFG['batch_size'], num_workers=4, pin_memory=True)

        model = OrthogonalGDIFNet(RISK_TYPES, input_map, **MODEL_CFG).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=TRAIN_CFG['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

        best_val_loss = float('inf')
        for epoch in range(TRAIN_CFG['epochs']):
            model.train()
            total_train_loss = 0
            for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_CFG['epochs']} [训练]"):
                data = data.reshape(-1, len(data_channels)).to(device)
                labels = labels.reshape(-1, len(RISK_TYPES)).to(device)
                data_dict = {name: data[:, i].unsqueeze(1) for i, name in enumerate(data_channels)}

                optimizer.zero_grad()
                risk_vector, latent_risks = model(data_dict, edge_index)
                main_loss = criterion(risk_vector, labels)
                ortho_loss = orthogonality_loss(latent_risks)
                total_loss = main_loss + TRAIN_CFG['ortho_loss_weight'] * ortho_loss
                total_loss.backward()
                optimizer.step()
                total_train_loss += total_loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for data, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{TRAIN_CFG['epochs']} [验证]"):
                    data = data.reshape(-1, len(data_channels)).to(device)
                    labels = labels.reshape(-1, len(RISK_TYPES)).to(device)
                    data_dict = {name: data[:, i].unsqueeze(1) for i, name in enumerate(data_channels)}
                    risk_vector, _ = model(data_dict, edge_index)
                    val_loss = criterion(risk_vector, labels)
                    total_val_loss += val_loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch+1} | 训练损失: {avg_train_loss:.5f} | 验证损失: {avg_val_loss:.5f}")
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_save_path = os.path.join(MODELS_DIR, f"{scenario_name}_model.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"  -> 验证损失下降，已保存新模型至: {model_save_path}")

if __name__ == '__main__':
    train_all_scenarios()
