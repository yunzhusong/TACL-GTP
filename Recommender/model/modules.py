import torch
from torch import nn as nn
from torch.nn import functional as F


# Pooling Modules
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attn_fc1 = nn.Linear(input_dim, hidden_dim//2)
        self.attn_fc2 = nn.Linear(hidden_dim//2, 1)
        self.drop_layer = nn.Dropout(p=0.2)

    def forward(self, x, attn_mask=None):
        batch_size = x.shape[0]

        e = self.attn_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.attn_fc2(e)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = F.softmax(alpha, dim=1)
        #alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (batch_size, -1))
        return x

class CTRRegressionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CTRRegressionHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.Tanh(),
            nn.Linear(hidden_dim//4, 1),
            nn.ReLU(),
        )

        self.input_dim = input_dim

    def forward(self, x):
        input_shape = x.shape
        x = self.head(x.view(-1, input_shape[-1]))
        x = x.view(input_shape[:2])
        return x

def initialize_weight(m):
  if isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)
