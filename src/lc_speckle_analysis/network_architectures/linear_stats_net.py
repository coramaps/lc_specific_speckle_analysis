"""
Linear Network Architecture for MeanAndStd Features

A minimal linear network for land cover classification using statistical features
(mean and standard deviation) from Sentinel-1 SAR data (VV and VH polarizations).
Input: 4 features [mean_VV, std_VV, mean_VH, std_VH]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LinearStatsNet(nn.Module):
    """
    Linear network for processing statistical features from SAR data.
    
    Architecture:
    - Input: 4 statistical features (mean and std for VV and VH channels)
    - Hidden Layer: Configurable size (default: 16, minimal parameters)
    - Output: Number of classes
    
    Features:
    - Dropout for regularization
    - Configurable activation function
    - Minimal parameter count for statistical feature processing
    """
    
    def __init__(self, 
                 num_classes: int,
                 input_size: int = 4,
                 hidden_size: int = 16,
                 dropout_rate: float = 0.2,
                 activation: str = 'relu'):
        """
        Initialize the linear statistics network.
        
        Args:
            num_classes: Number of output classes
            input_size: Number of input features (4 for meanandstd, 2 for std)
            hidden_size: Size of hidden layer (default: 16 for minimal params)
            dropout_rate: Dropout probability
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(LinearStatsNet, self).__init__()
        
        # Variable input size for statistical features
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Network layers
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"LinearStatsNet initialized:")
        logger.info(f"  - Input features: {self.input_size}")
        logger.info(f"  - Hidden size: {hidden_size}")
        logger.info(f"  - Output classes: {num_classes}")
        logger.info(f"  - Dropout rate: {dropout_rate}")
        logger.info(f"  - Activation: {activation}")
        logger.info(f"  - Total parameters: {self.count_parameters()}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size) with statistical features
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Input should be (batch_size, input_size)
        if x.dim() > 2:
            # If input has spatial dimensions, it should be already flattened
            raise ValueError(f"Expected 2D input (batch_size, {self.input_size}), got shape {x.shape}")
        
        if x.size(1) != self.input_size:
            raise ValueError(f"Expected {self.input_size} input features, got {x.size(1)}")
        
        # Forward pass
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_feature_importance(self, device=None):
        """
        Get feature importance based on first layer weights.
        
        Returns:
            Dict with feature names and their importance scores
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Get weights from first layer (input to hidden)
        weights = self.fc1.weight.data.cpu().numpy()  # Shape: (hidden_size, input_size)
        
        # Compute importance as mean absolute weight for each input feature
        feature_importance = np.mean(np.abs(weights), axis=0)
        
        # Generate feature names based on input size
        if self.input_size == 4:
            feature_names = ['Mean VV', 'Std VV', 'Mean VH', 'Std VH']
        elif self.input_size == 2:
            # Could be either std-only or mean-only, make generic
            feature_names = ['Feature VV', 'Feature VH']
        else:
            feature_names = [f'Feature_{i}' for i in range(self.input_size)]
        
        return dict(zip(feature_names, feature_importance))


def create_network(config):
    """
    Create LinearStatsNet instance from configuration.
    
    Args:
        config: Configuration object with neural network parameters
        
    Returns:
        LinearStatsNet instance
    """
    network = LinearStatsNet(
        num_classes=len(config.classes),
        hidden_size=config.neural_network.layer_sizes[0] if config.neural_network.layer_sizes else 16,
        dropout_rate=config.neural_network.dropout_rate,
        activation=config.neural_network.activation_function
    )
    
    return network


def plot_feature_importance(network, save_path=None):
    """
    Plot feature importance from the trained network.
    
    Args:
        network: Trained LinearStatsNet instance
        save_path: Optional path to save the plot
    """
    importance_dict = network.get_feature_importance()
    
    features = list(importance_dict.keys())
    importance = list(importance_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(features, importance, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title('Feature Importance in LinearStatsNet', fontsize=14, fontweight='bold')
    plt.ylabel('Mean Absolute Weight', fontsize=12)
    plt.xlabel('Statistical Features', fontsize=12)
    
    # Add value labels on bars
    for bar, value in zip(bars, importance):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to: {save_path}")
    
    plt.show()


def test_network():
    """Test the LinearStatsNet with dummy data."""
    # Test parameters
    batch_size = 32
    num_classes = 4
    
    # Create network
    network = LinearStatsNet(num_classes=num_classes, hidden_size=16)
    
    # Create dummy input data (statistical features)
    dummy_input = torch.randn(batch_size, 4)  # 4 statistical features
    
    # Test forward pass
    output = network(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {network.count_parameters()}")
    
    # Test feature importance
    importance = network.get_feature_importance()
    print("Feature importance:")
    for feature, imp in importance.items():
        print(f"  {feature}: {imp:.4f}")


if __name__ == "__main__":
    test_network()
