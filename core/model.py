import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class RiskEncoder(nn.Module):
    def __init__(self, num_input_modalities, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(num_input_modalities, 64)
        self.fc2 = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

class OrthogonalGDIFNet(nn.Module):
    def __init__(self, risk_types, input_map, latent_dim, gnn_layers, gat_heads):
        super().__init__()
        self.risk_types = risk_types
        self.input_map = input_map
        self.latent_dim = latent_dim

        self.encoders = nn.ModuleDict()
        for risk in self.risk_types:
            num_inputs = len(self.input_map[risk])
            self.encoders[risk] = RiskEncoder(num_inputs, self.latent_dim)

        self.gnn = GATv2Conv(self.latent_dim, self.latent_dim, heads=gat_heads, concat=False)

        self.risk_heads = nn.ModuleDict()
        for risk in self.risk_types:
            self.risk_heads[risk] = nn.Sequential(
                nn.Linear(self.latent_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

    def forward(self, multi_modal_data, edge_index):
        latent_risks = {}
        for risk in self.risk_types:
            input_modalities = [multi_modal_data[mod_name] for mod_name in self.input_map[risk]]
            risk_input_tensor = torch.cat(input_modalities, dim=1)
            latent_risks[risk] = self.encoders[risk](risk_input_tensor)

        risk_predictions = []
        for risk in self.risk_types:
            gnn_output = F.gelu(self.gnn(latent_risks[risk], edge_index))
            prediction = self.risk_heads[risk](gnn_output)
            risk_predictions.append(prediction)
        
        risk_vector = torch.cat(risk_predictions, dim=1)
        
        return risk_vector, latent_risks
