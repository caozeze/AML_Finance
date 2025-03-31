import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=3, hidden_dims=[128, 64], dropout=0.1):
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 构建MLP层
        layers = []
        prev_dim = input_dim
        
        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 添加输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # 将所有层组合成一个序列
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        # 直接通过MLP网络
        return self.model(x)
    
    def get_attention_weights(self, x):
        """为了与TabTransformer接口兼容，返回None
        Args:
            x: 输入张量 [batch_size, input_dim]
        Returns:
            None，因为MLP没有注意力机制
        """
        return None