import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """Residual block for the neural network"""
    
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x

class ChessNet(nn.Module):
    """Neural network for Chess with policy and value heads"""
    
    def __init__(self, input_channels=20, residual_blocks=10, filters=256, hidden_units=256):
        super(ChessNet, self).__init__()
        
        self.input_channels = input_channels
        self.residual_blocks = residual_blocks
        self.filters = filters
        self.hidden_units = hidden_units
        
        self.conv_input = nn.Conv2d(input_channels, filters, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(filters)
        
        self.residual_layers = nn.ModuleList(
            [ResidualBlock(filters) for _ in range(residual_blocks)]
        )
        
        self.policy_conv = nn.Conv2d(filters, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, hidden_units)
        self.policy_head = nn.Linear(hidden_units, 4672)
        
        self.value_conv = nn.Conv2d(filters, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, hidden_units)
        self.value_fc2 = nn.Linear(hidden_units, 1)
        
    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        for residual in self.residual_layers:
            x = residual(x)
        
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = F.relu(self.policy_fc(policy))
        policy = self.policy_head(policy)
        policy = F.log_softmax(policy, dim=1)
        
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, board_tensor):
        self.eval()
        with torch.no_grad():
            board_tensor = torch.FloatTensor(board_tensor).unsqueeze(0)
            policy, value = self.forward(board_tensor)
            policy = torch.exp(policy).squeeze(0).cpu().numpy()
            value = value.squeeze(0).cpu().numpy()[0]
        return policy, value
