import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, input_dim, num_classes=3, num_heads=4, num_layers=2, dim_model=64, dim_ff=128, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dim_model = dim_model
        
        # Create Feature Embeddings
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, dim_model) for _ in range(input_dim)
        ])
        
        # Add Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_dim, dim_model))
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(dim_model * input_dim, dim_ff), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Convert feature to sequence representation [batch_size, seq_len, dim_model]
        feature_embeddings = []
        for i in range(self.input_dim):
            # Extract each feature and convert to [batch_size, 1]
            feature = x[:, i].unsqueeze(1) 
            embedded = self.feature_embeddings[i](feature)  # [batch_size, dim_model]
            feature_embeddings.append(embedded)
        
        # Stack all feature embeddings to form a sequence [batch_size, input_dim, dim_model]
        x = torch.stack(feature_embeddings, dim=1)  # [batch_size, input_dim, dim_model]
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Transformer Encoder
        x = self.transformer_encoder(x)  # [batch_size, input_dim, dim_model]
        
        # Flatten the output for classification
        x = x.reshape(batch_size, -1)  # [batch_size, input_dim * dim_model]
        
        # Classifier
        logits = self.classifier(x)
        
        return logits

    def get_attention_weights(self, x):
        batch_size = x.shape[0]
        
        # Convert Feature to Sequence Representation
        feature_embeddings = []
        for i in range(self.input_dim):
            feature = x[:, i].unsqueeze(1)
            embedded = self.feature_embeddings[i](feature)
            feature_embeddings.append(embedded)
        
        x = torch.stack(feature_embeddings, dim=1)
        x = x + self.positional_encoding
        
        # Get Attention Weights
        attention_weights = []
        for layer in self.transformer_encoder.layers:
            attn_output, attn_weights = layer.self_attn(
                x, x, x,
                need_weights=True
            )
            attention_weights.append(attn_weights)
            # Update x for next layer
            x = layer(x)
        
        # Return Last Attention Layer [batch_size, num_heads, input_dim, input_dim]
        return attention_weights[-1]