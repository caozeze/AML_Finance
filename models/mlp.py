import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=3, hidden_dims=[128, 64], dropout=0.1):
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Construct the MLP layers
        layers = []
        prev_dim = input_dim
        
        # Add Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # Combine all layers into a Sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
    def get_attention_weights(self, x):
        return None