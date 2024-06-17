"""Convolutional neural network module"""
import torch
from torch import nn


class CNN(nn.Module):
    """Simple CNN for model testing"""

    def __init__(
        self,
        filters: int,
        output: int,
    ):
        super().__init__()

        self.simple_cnn = nn.Sequential(
            nn.Conv3d(
                1,
                filters,
                4,
            ),
            nn.ReLU(),
            nn.Conv3d(
                filters,
                filters * 2,
                4,
            ),
            nn.ReLU(),
            nn.Conv3d(
                filters * 2,
                filters * 4,
                4,
            ),
            nn.ReLU(),
            nn.Conv3d(
                filters * 4,
                filters * 8,
                4,
            ),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*122*122*122, output),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the network

        Args:
            x (torch.Tensor): input tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: output tensor for regression.
        """
        x = self.simple_cnn(x)

        return self.regression_head(x)