"""
3D U-Net Architecture for TripoSR Mesh Fusion

This module implements a 3D U-Net model that takes multiple voxelized 3D meshes
(generated from X-ray images at different angles via TripoSR) and fuses them
into a single coherent bone model.

Architecture Overview:
- Input: Stacked voxel grids from N TripoSR reconstructions (N x D x H x W)
- Encoder: 4 downsampling blocks with 3D convolutions
- Bottleneck: Dense feature representation
- Decoder: 4 upsampling blocks with skip connections
- Output: Single fused voxel grid (1 x D x H x W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class ConvBlock3D(nn.Module):
    """
    3D Convolutional block with optional residual connection.
    
    Components: Conv3D -> BatchNorm -> ReLU -> Conv3D -> BatchNorm -> ReLU
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_residual: bool = False,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.use_residual = use_residual
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Residual projection if channel dimensions don't match
        if use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_residual:
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
            out = out + residual
        
        out = self.relu(out)
        return out


class EncoderBlock3D(nn.Module):
    """
    Encoder block: ConvBlock followed by MaxPool for downsampling.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_residual: bool = False,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.conv_block = ConvBlock3D(
            in_channels, out_channels, 
            use_residual=use_residual,
            dropout_rate=dropout_rate
        )
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (pooled output, skip connection features)"""
        features = self.conv_block(x)
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock3D(nn.Module):
    """
    Decoder block: Upsample -> Concatenate skip -> ConvBlock
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_residual: bool = False,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels, in_channels // 2, 
            kernel_size=2, stride=2
        )
        self.conv_block = ConvBlock3D(
            in_channels // 2 + skip_channels, out_channels,
            use_residual=use_residual,
            dropout_rate=dropout_rate
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # Handle size mismatches due to odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class AttentionGate3D(nn.Module):
    """
    Attention gate for focusing on relevant spatial regions.
    Used in attention U-Net variant.
    """
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.W_g = nn.Conv3d(gate_channels, inter_channels, kernel_size=1)
        self.W_x = nn.Conv3d(skip_channels, inter_channels, kernel_size=1, stride=2)
        self.psi = nn.Conv3d(inter_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gate signal from decoder (lower resolution)
            x: Skip connection from encoder (higher resolution)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Align sizes
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        psi = self.upsample(psi)
        
        # Handle size mismatch
        if psi.shape[2:] != x.shape[2:]:
            psi = F.interpolate(psi, size=x.shape[2:], mode='trilinear', align_corners=True)
        
        return x * psi


class UNet3D(nn.Module):
    """
    3D U-Net for fusing multiple TripoSR mesh reconstructions.
    
    Args:
        in_channels: Number of input channels (typically N views from TripoSR)
        out_channels: Number of output channels (typically 1 for single fused mesh)
        base_features: Number of features in first encoder layer
        depth: Number of encoder/decoder levels
        use_residual: Whether to use residual connections in conv blocks
        use_attention: Whether to use attention gates in skip connections
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(
        self,
        in_channels: int = 3,  # 3 TripoSR views
        out_channels: int = 1,  # Single fused output
        base_features: int = 32,
        depth: int = 4,
        use_residual: bool = True,
        use_attention: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.depth = depth
        self.use_attention = use_attention
        
        # Calculate feature sizes for each level
        features = [base_features * (2 ** i) for i in range(depth + 1)]
        
        # Input convolution
        self.input_conv = ConvBlock3D(
            in_channels, features[0],
            use_residual=use_residual,
            dropout_rate=dropout_rate
        )
        
        # Encoder path
        self.encoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(
                EncoderBlock3D(
                    features[i], features[i + 1],
                    use_residual=use_residual,
                    dropout_rate=dropout_rate
                )
            )
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(
            features[-1], features[-1] * 2,
            use_residual=use_residual,
            dropout_rate=dropout_rate
        )
        
        # Attention gates (if used)
        # Gate signal comes from decoder path, skip from encoder
        if use_attention:
            self.attention_gates = nn.ModuleList()
            # First attention gate: input is bottleneck output (features[-1]*2)
            # Subsequent: input is previous decoder output
            decoder_channels = features[-1] * 2  # Start with bottleneck output
            for i in range(depth):
                # Skip connections after reverse: features[depth], features[depth-1], ...
                skip_channels = features[depth - i]  # Encoder skip channels
                inter_channels = max(skip_channels // 2, 16)
                self.attention_gates.append(
                    AttentionGate3D(decoder_channels, skip_channels, inter_channels)
                )
                # After decoder, channels become skip_channels (decoder output)
                decoder_channels = skip_channels
        
        # Decoder path
        self.decoders = nn.ModuleList()
        decoder_in = features[-1] * 2
        for i in range(depth):
            # Skip connections after reverse: features[depth], features[depth-1], ...
            skip_channels = features[depth - i]
            out_features = skip_channels
            self.decoders.append(
                DecoderBlock3D(
                    decoder_in, skip_channels, out_features,
                    use_residual=use_residual,
                    dropout_rate=dropout_rate
                )
            )
            decoder_in = out_features
        
        # Final decoder output has features[1] channels
        final_channels = features[1]
        
        # Output convolution
        self.output_conv = nn.Sequential(
            nn.Conv3d(final_channels, final_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(final_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(final_channels // 2, out_channels, kernel_size=1),
            nn.Sigmoid()  # Output occupancy probability [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N_views, D, H, W)
               where N_views is the number of TripoSR reconstructions
        
        Returns:
            Fused output of shape (B, 1, D, H, W)
        """
        # Initial convolution
        x = self.input_conv(x)
        
        # Encoder path - save skip connections
        skip_connections = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for i, decoder in enumerate(self.decoders):
            skip = skip_connections[i]
            
            # Apply attention gate if enabled
            if self.use_attention:
                skip = self.attention_gates[i](x, skip)
            
            x = decoder(x, skip)
        
        # Output
        return self.output_conv(x)


class UNet3DMultiScale(nn.Module):
    """
    Multi-scale 3D U-Net with deep supervision for improved training.
    
    This variant outputs predictions at multiple resolutions during training
    to provide additional gradient signals.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_features: int = 32,
        depth: int = 4,
        use_residual: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.depth = depth
        features = [base_features * (2 ** i) for i in range(depth + 1)]
        
        # Main U-Net components
        self.input_conv = ConvBlock3D(in_channels, features[0], use_residual=use_residual)
        
        self.encoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(
                EncoderBlock3D(features[i], features[i + 1], use_residual=use_residual)
            )
        
        self.bottleneck = ConvBlock3D(
            features[-1], features[-1] * 2, use_residual=use_residual
        )
        
        self.decoders = nn.ModuleList()
        decoder_in = features[-1] * 2
        for i in range(depth):
            skip_channels = features[depth - 1 - i]
            out_features = skip_channels
            self.decoders.append(
                DecoderBlock3D(decoder_in, skip_channels, out_features, use_residual=use_residual)
            )
            decoder_in = out_features
        
        # Deep supervision outputs
        self.deep_outputs = nn.ModuleList()
        for i in range(depth):
            feat_channels = features[depth - 1 - i]
            self.deep_outputs.append(
                nn.Conv3d(feat_channels, out_channels, kernel_size=1)
            )
        
        self.output_conv = nn.Sequential(
            nn.Conv3d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, return_deep: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            return_deep: If True, return deep supervision outputs (for training)
        """
        x = self.input_conv(x)
        
        skip_connections = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)
        
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        deep_outputs = []
        
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])
            if return_deep:
                deep_out = torch.sigmoid(self.deep_outputs[i](x))
                deep_outputs.append(deep_out)
        
        output = self.output_conv(x)
        
        if return_deep:
            return output, deep_outputs
        return output


def get_model(
    model_type: str = 'standard',
    in_channels: int = 3,
    out_channels: int = 1,
    base_features: int = 32,
    depth: int = 4,
    **kwargs
) -> nn.Module:
    """
    Factory function to create 3D U-Net models.
    
    Args:
        model_type: 'standard', 'attention', or 'multiscale'
        in_channels: Number of input TripoSR views
        out_channels: Number of output channels
        base_features: Base feature count
        depth: Network depth
    """
    if model_type == 'standard':
        return UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=base_features,
            depth=depth,
            use_attention=False,
            **kwargs
        )
    elif model_type == 'attention':
        return UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=base_features,
            depth=depth,
            use_attention=True,
            **kwargs
        )
    elif model_type == 'multiscale':
        return UNet3DMultiScale(
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=base_features,
            depth=depth,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test model creation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = UNet3D(
        in_channels=3,  # 3 TripoSR views
        out_channels=1,
        base_features=32,
        depth=4
    ).to(device)
    
    # Test input (batch=2, views=3, depth=64, height=64, width=64)
    x = torch.randn(2, 3, 64, 64, 64).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
