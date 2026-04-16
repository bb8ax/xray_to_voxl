"""
Loss Functions for 3D Mesh Fusion Training

This module provides various loss functions suitable for voxel-based
3D reconstruction and mesh fusion tasks.

Implemented losses:
- Dice Loss (for handling class imbalance)
- Focal Loss (for hard example mining)
- Binary Cross-Entropy with Dice
- Boundary-aware Loss
- Chamfer Distance (mesh-based)
- Combined losses with Chamfer Distance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    Good for handling class imbalance in voxel occupancy.
    
    Dice = 2 * |A  &  B| / (|A| + |B|)
    Loss = 1 - Dice
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, 1, D, H, W), values in [0, 1]
            target: Ground truth (B, 1, D, H, W), values in {0, 1}
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss and Binary Cross-Entropy.
    
    Provides both region-based (Dice) and pixel-wise (BCE) supervision.
    """
    
    def __init__(
        self, 
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1e-6
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        return self.dice_weight * dice + self.bce_weight * bce


class FocalLoss(nn.Module):
    """
    Focal Loss for hard example mining.
    
    FL(p) = -alpha(1-p)^gamma * log(p)
    
    Focuses learning on hard examples by down-weighting easy ones.
    """
    
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        
        # Binary cross entropy
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        # Focal weight
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice that allows tuning
    false positive vs false negative penalty.
    
    Tversky = TP / (TP + alpha*FP + beta*FN)
    
    alpha = beta = 0.5 gives Dice loss
    alpha = beta = 1.0 gives Jaccard loss
    """
    
    def __init__(
        self, 
        alpha: float = 0.5, 
        beta: float = 0.5,
        smooth: float = 1e-6
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        
        return 1.0 - tversky


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that emphasizes voxel boundaries.
    
    Uses Sobel-like filters to detect boundaries and applies
    higher weight to boundary regions.
    """
    
    def __init__(self, boundary_weight: float = 2.0):
        super().__init__()
        self.boundary_weight = boundary_weight
        
        # 3D Sobel-like kernels for boundary detection
        self.register_buffer('sobel_x', self._create_sobel_kernel(0))
        self.register_buffer('sobel_y', self._create_sobel_kernel(1))
        self.register_buffer('sobel_z', self._create_sobel_kernel(2))
    
    def _create_sobel_kernel(self, axis: int) -> torch.Tensor:
        """Create a 3D Sobel kernel for the given axis."""
        kernel = torch.zeros(1, 1, 3, 3, 3)
        
        if axis == 0:  # X gradient
            kernel[0, 0, :, 1, 1] = torch.tensor([-1, 0, 1])
        elif axis == 1:  # Y gradient
            kernel[0, 0, 1, :, 1] = torch.tensor([-1, 0, 1])
        else:  # Z gradient
            kernel[0, 0, 1, 1, :] = torch.tensor([-1, 0, 1])
        
        return kernel
    
    def _compute_boundary(self, x: torch.Tensor) -> torch.Tensor:
        """Compute boundary magnitude using gradient."""
        gx = F.conv3d(x, self.sobel_x, padding=1)
        gy = F.conv3d(x, self.sobel_y, padding=1)
        gz = F.conv3d(x, self.sobel_z, padding=1)
        
        gradient_magnitude = torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-8)
        return gradient_magnitude
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute boundaries
        target_boundary = self._compute_boundary(target)
        
        # Create weight map (higher weight at boundaries)
        weight = 1.0 + (self.boundary_weight - 1.0) * (target_boundary > 0.1).float()
        
        # Weighted BCE
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        weighted_bce = (bce * weight).mean()
        
        return weighted_bce


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss for deep supervision in U-Net.
    
    Applies loss at multiple resolutions with decreasing weights.
    """
    
    def __init__(
        self, 
        base_loss: nn.Module,
        scales: int = 4,
        weight_decay: float = 0.5
    ):
        super().__init__()
        self.base_loss = base_loss
        self.scales = scales
        
        # Compute weights: [1.0, 0.5, 0.25, 0.125, ...]
        self.weights = [weight_decay ** i for i in range(scales)]
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def forward(
        self, 
        pred: torch.Tensor, 
        deep_outputs: list, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Main output prediction
            deep_outputs: List of intermediate predictions (low to high resolution)
            target: Ground truth at full resolution
        """
        # Main loss
        total_loss = self.weights[0] * self.base_loss(pred, target)
        
        # Deep supervision losses
        for i, deep_pred in enumerate(deep_outputs):
            # Downsample target to match prediction size
            target_size = deep_pred.shape[2:]
            target_down = F.interpolate(
                target, size=target_size, mode='trilinear', align_corners=True
            )
            target_down = (target_down > 0.5).float()  # Re-binarize
            
            total_loss += self.weights[i + 1] * self.base_loss(deep_pred, target_down)
        
        return total_loss


class ChamferDistanceLoss(nn.Module):
    """
    Chamfer Distance loss for comparing 3D shapes.
    
    Converts voxel grids to point clouds and computes bidirectional
    nearest neighbor distances.
    
    CD(A, B) = (1/|A|) * sum_a min_b ||a - b||^2 + (1/|B|) * sum_b min_a ||b - a||^2
    
    Args:
        threshold: Threshold for converting voxels to points
        normalize: Whether to normalize point clouds
        bidirectional: If True, compute both directions; if False, only pred->target
    """
    
    def __init__(
        self, 
        threshold: float = 0.5,
        normalize: bool = True,
        bidirectional: bool = True,
        max_points: int = 10000
    ):
        super().__init__()
        self.threshold = threshold
        self.normalize = normalize
        self.bidirectional = bidirectional
        self.max_points = max_points
    
    def _voxels_to_points(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        Convert voxel grid to point cloud.
        
        Args:
            voxels: (D, H, W) tensor
            
        Returns:
            (N, 3) tensor of point coordinates
        """
        # Get indices of occupied voxels
        indices = torch.nonzero(voxels > self.threshold, as_tuple=False).float()
        
        # Subsample if too many points
        if indices.shape[0] > self.max_points:
            perm = torch.randperm(indices.shape[0], device=indices.device)[:self.max_points]
            indices = indices[perm]
        
        # Normalize to [-1, 1]
        if self.normalize and indices.shape[0] > 0:
            indices = indices / (voxels.shape[0] - 1) * 2 - 1
        
        return indices
    
    def _chamfer_distance_single(
        self, 
        points1: torch.Tensor, 
        points2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Chamfer distance between two point clouds.
        
        Args:
            points1: (N, 3) tensor
            points2: (M, 3) tensor
            
        Returns:
            Scalar Chamfer distance
        """
        if points1.shape[0] == 0 or points2.shape[0] == 0:
            return torch.tensor(0.0, device=points1.device, requires_grad=True)
        
        # Compute pairwise distances: (N, M)
        # Using efficient computation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        p1_norm = (points1 ** 2).sum(dim=1, keepdim=True)  # (N, 1)
        p2_norm = (points2 ** 2).sum(dim=1, keepdim=True)  # (M, 1)
        
        # (N, M) = (N, 1) + (1, M) - 2*(N, M)
        dist_matrix = p1_norm + p2_norm.t() - 2.0 * torch.mm(points1, points2.t())
        dist_matrix = torch.clamp(dist_matrix, min=0.0)  # Numerical stability
        
        # Min distance from each point in p1 to p2
        min_dist_1to2 = dist_matrix.min(dim=1)[0].mean()
        
        if self.bidirectional:
            # Min distance from each point in p2 to p1
            min_dist_2to1 = dist_matrix.min(dim=0)[0].mean()
            return min_dist_1to2 + min_dist_2to1
        
        return min_dist_1to2
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted voxels (B, 1, D, H, W)
            target: Target voxels (B, 1, D, H, W)
            
        Returns:
            Chamfer distance loss
        """
        batch_size = pred.shape[0]
        total_loss = 0.0
        valid_samples = 0
        
        for b in range(batch_size):
            pred_points = self._voxels_to_points(pred[b, 0])
            target_points = self._voxels_to_points(target[b, 0])
            
            if pred_points.shape[0] > 0 and target_points.shape[0] > 0:
                cd = self._chamfer_distance_single(pred_points, target_points)
                total_loss = total_loss + cd
                valid_samples += 1
        
        if valid_samples == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        return total_loss / valid_samples


class ChamferDiceLoss(nn.Module):
    """
    Combined Chamfer Distance and Dice Loss.
    
    Provides both point-cloud based (Chamfer) and voxel-based (Dice) supervision.
    """
    
    def __init__(
        self,
        chamfer_weight: float = 0.5,
        dice_weight: float = 0.5,
        threshold: float = 0.5
    ):
        super().__init__()
        self.chamfer_weight = chamfer_weight
        self.dice_weight = dice_weight
        self.chamfer_loss = ChamferDistanceLoss(threshold=threshold)
        self.dice_loss = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        chamfer = self.chamfer_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        return self.chamfer_weight * chamfer + self.dice_weight * dice


class ChamferBCELoss(nn.Module):
    """
    Combined Chamfer Distance, Dice, and BCE Loss.
    """
    
    def __init__(
        self,
        chamfer_weight: float = 0.3,
        dice_weight: float = 0.4,
        bce_weight: float = 0.3,
        threshold: float = 0.5
    ):
        super().__init__()
        self.chamfer_weight = chamfer_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.chamfer_loss = ChamferDistanceLoss(threshold=threshold)
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        chamfer = self.chamfer_loss(pred, target)
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        return self.chamfer_weight * chamfer + self.dice_weight * dice + self.bce_weight * bce


class IoULoss(nn.Module):
    """
    Intersection over Union (Jaccard) Loss.
    
    IoU = |A  &  B| / |A  |  B|
    Loss = 1 - IoU
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - iou


class CombinedLoss(nn.Module):
    """
    Flexible combined loss allowing any combination of losses.
    """
    
    def __init__(self, losses: dict, weights: Optional[dict] = None):
        """
        Args:
            losses: Dict mapping loss names to loss modules
            weights: Dict mapping loss names to weights (default: equal weights)
        """
        super().__init__()
        
        self.losses = nn.ModuleDict(losses)
        
        if weights is None:
            weights = {name: 1.0 for name in losses}
        self.weights = weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        
        for name, loss_fn in self.losses.items():
            loss_val = loss_fn(pred, target)
            total_loss += self.weights[name] * loss_val
        
        return total_loss


def get_loss_function(
    loss_type: str = 'dice_bce',
    **kwargs
) -> nn.Module:
    """
    Factory function for creating loss functions.
    
    Args:
        loss_type: One of 'dice', 'bce', 'dice_bce', 'focal', 'tversky',
                   'boundary', 'iou', 'chamfer', 'chamfer_dice', 'chamfer_bce',
                   'combined'
    """
    if loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'bce':
        return nn.BCELoss()
    elif loss_type == 'dice_bce':
        return DiceBCELoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_type == 'boundary':
        return BoundaryLoss(**kwargs)
    elif loss_type == 'iou':
        return IoULoss(**kwargs)
    elif loss_type == 'chamfer':
        return ChamferDistanceLoss(**kwargs)
    elif loss_type == 'chamfer_dice':
        return ChamferDiceLoss(**kwargs)
    elif loss_type == 'chamfer_bce':
        return ChamferBCELoss(**kwargs)
    elif loss_type == 'combined':
        # Default combined loss
        return CombinedLoss(
            losses={
                'dice': DiceLoss(),
                'bce': nn.BCELoss(),
                'iou': IoULoss()
            },
            weights={
                'dice': 1.0,
                'bce': 0.5,
                'iou': 0.5
            }
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == '__main__':
    # Test losses
    print("Testing loss functions...")
    
    # Create dummy predictions and targets
    pred = torch.sigmoid(torch.randn(2, 1, 32, 32, 32))
    target = (torch.rand(2, 1, 32, 32, 32) > 0.7).float()
    
    # Test each loss
    losses = ['dice', 'bce', 'dice_bce', 'focal', 'tversky', 'boundary', 'iou', 
              'chamfer', 'chamfer_dice', 'chamfer_bce']
    
    for loss_type in losses:
        loss_fn = get_loss_function(loss_type)
        loss_val = loss_fn(pred, target)
        print(f"{loss_type}: {loss_val.item():.4f}")
