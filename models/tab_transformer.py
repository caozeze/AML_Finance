import torch
import torch.nn as nn
import torch.nn.functional as F

class TabTransformer(nn.Module):
    def __init__(self, input_dim, num_classes=3, num_heads=4, num_layers=2, dim_model=64, dim_ff=128, dropout=0.1):
        super(TabTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dim_model = dim_model
        
        # 特征嵌入层
        self.feature_embedding = nn.Linear(input_dim, dim_model)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, num_classes)
        )
    
    def forward(self, x):
        # 特征嵌入
        x = self.feature_embedding(x)

        # 添加序列维度
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, dim_model]
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 取序列的平均值作为特征表示
        x = torch.mean(x, dim=1)
        
        # 分类
        logits = self.classifier(x)
        
        return logits
        
    def get_attention_weights(self, x):
        """
        获取Transformer编码器的注意力权重
        Args:
            x: 输入张量 [batch_size, input_dim]
        Returns:
            注意力权重张量 [batch_size, num_heads, seq_len, seq_len]
        """
        # 特征嵌入
        x = self.feature_embedding(x)
        
        # 添加序列维度
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, dim_model]
        
        # 获取所有编码器层的注意力权重
        attention_weights = []
        for layer in self.transformer_encoder.layers:
            # 通过self_attn获取注意力权重
            attn_output, attn_weights = layer.self_attn(
                x, x, x,
                need_weights=True
            )
            attention_weights.append(attn_weights)
        
        # 返回最后一层的注意力权重
        return attention_weights[-1]