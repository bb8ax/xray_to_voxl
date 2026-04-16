"""
Evaluation Metrics for 3D Mesh Fusion

This module provides evaluation metrics for comparing 3D voxel predictions
with ground truth:

- Chamfer Distance: Point cloud based distance metric
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index (3D version)
- RMSE: Root Mean Square Error
- Dice Coefficient
- IoU (Jaccard Index)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union


def compute_chamfer_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    max_points: int = 10000,
    normalize: bool = True
) -> float:
    """
    Compute Chamfer Distance between predicted and target voxel grids.
    
    Chamfer Distance measures the average nearest neighbor distance between
    two point clouds (derived from voxel grids).
    
    CD(A, B) = (1/|A|) * sum_a min_b ||a-b||^2 + (1/|B|) * sum_b min_a ||b-a||^2
    
    Args:
        pred: Predicted voxels (B, 1, D, H, W) or (D, H, W)
        target: Target voxels (B, 1, D, H, W) or (D, H, W)
        threshold: Threshold for converting voxels to points
        max_points: Maximum points to use (for memory efficiency)
        normalize: Whether to normalize point coordinates
        
    Returns:
        Chamfer distance (lower is better)
    """
    # Handle different input shapes
    if pred.dim() == 5:
        pred = pred.squeeze(1)  # (B, D, H, W)
    if target.dim() == 5:
        target = target.squeeze(1)
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)  # (1, D, H, W)
        target = target.unsqueeze(0)
    
    batch_size = pred.shape[0]
    total_cd = 0.0
    valid_count = 0
    
    for b in range(batch_size):
        # Convert to point clouds
        pred_points = _voxels_to_points(pred[b], threshold, max_points, normalize)
        target_points = _voxels_to_points(target[b], threshold, max_points, normalize)
        
        if pred_points.shape[0] == 0 or target_points.shape[0] == 0:
            continue
        
        # Compute bidirectional Chamfer distance
        cd = _chamfer_distance_single(pred_points, target_points)
        total_cd += cd.item()
        valid_count += 1
    
    if valid_count == 0:
        return 0.0
    
    return total_cd / valid_count


def _voxels_to_points(
    voxels: torch.Tensor, 
    threshold: float = 0.5,
    max_points: int = 10000,
    normalize: bool = True
) -> torch.Tensor:
    """Convert voxel grid to point cloud."""
    indices = torch.nonzero(voxels > threshold, as_tuple=False).float()
    
    if indices.shape[0] > max_points:
        perm = torch.randperm(indices.shape[0], device=indices.device)[:max_points]
        indices = indices[perm]
    
    if normalize and indices.shape[0] > 0:
        indices = indices / (voxels.shape[0] - 1) * 2 - 1
    
    return indices


def _chamfer_distance_single(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    """Compute Chamfer distance between two point clouds."""
    # Efficient pairwise distance computation
    p1_norm = (points1 ** 2).sum(dim=1, keepdim=True)
    p2_norm = (points2 ** 2).sum(dim=1, keepdim=True)
    dist_matrix = p1_norm + p2_norm.t() - 2.0 * torch.mm(points1, points2.t())
    dist_matrix = torch.clamp(dist_matrix, min=0.0)
    
    # Bidirectional nearest neighbor distances
    min_dist_1to2 = dist_matrix.min(dim=1)[0].mean()
    min_dist_2to1 = dist_matrix.min(dim=0)[0].mean()
    
    return min_dist_1to2 + min_dist_2to1


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    PSNR = 10 * log10(MAX^2 / MSE)
    
    Higher PSNR indicates better quality.
    
    Args:
        pred: Predicted voxels (any shape)
        target: Target voxels (same shape as pred)
        max_val: Maximum possible value (1.0 for normalized voxels)
        
    Returns:
        PSNR in dB (higher is better)
    """
    mse = F.mse_loss(pred, target).item()
    
    if mse == 0:
        return float('inf')
    
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr


def compute_ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2
) -> float:
    """
    Compute 3D Structural Similarity Index (SSIM).
    
    SSIM compares local patterns of pixel intensities that have been
    normalized for luminance and contrast.
    
    SSIM(x, y) = (2*mux*muy + C1)(2*sigmaxy + C2) / ((mux^2 + muy^2 + C1)(sigmax^2 + sigmay^2 + C2))
    
    Args:
        pred: Predicted voxels (B, 1, D, H, W) or (D, H, W)
        target: Target voxels (same shape as pred)
        window_size: Size of the Gaussian window
        C1, C2: Stability constants
        
    Returns:
        SSIM value in [0, 1] (higher is better)
    """
    # Handle different input shapes
    if pred.dim() == 3:
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif pred.dim() == 4:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
    
    # Create 3D Gaussian window
    window = _create_3d_gaussian_window(window_size, 1, pred.device)
    
    # Compute local means
    mu_pred = F.conv3d(pred, window, padding=window_size // 2)
    mu_target = F.conv3d(target, window, padding=window_size // 2)
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    # Compute local variances and covariance
    sigma_pred_sq = F.conv3d(pred ** 2, window, padding=window_size // 2) - mu_pred_sq
    sigma_target_sq = F.conv3d(target ** 2, window, padding=window_size // 2) - mu_target_sq
    sigma_pred_target = F.conv3d(pred * target, window, padding=window_size // 2) - mu_pred_target
    
    # Clamp variances to avoid negative values due to numerical errors
    sigma_pred_sq = torch.clamp(sigma_pred_sq, min=0)
    sigma_target_sq = torch.clamp(sigma_target_sq, min=0)
    
    # Compute SSIM
    numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    
    ssim_map = numerator / (denominator + 1e-8)
    
    return ssim_map.mean().item()


def _create_3d_gaussian_window(window_size: int, channels: int, device: torch.device) -> torch.Tensor:
    """Create a 3D Gaussian window for SSIM computation."""
    sigma = 1.5
    
    # 1D Gaussian
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    gauss_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    
    # 3D Gaussian (outer product)
    gauss_3d = gauss_1d.view(-1, 1, 1) * gauss_1d.view(1, -1, 1) * gauss_1d.view(1, 1, -1)
    gauss_3d = gauss_3d / gauss_3d.sum()
    
    # Reshape for conv3d: (out_channels, in_channels, D, H, W)
    window = gauss_3d.unsqueeze(0).unsqueeze(0)
    window = window.expand(channels, 1, window_size, window_size, window_size).contiguous()
    
    return window


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Root Mean Square Error (RMSE).
    
    RMSE = sqrt(mean((pred - target)^2))
    
    Args:
        pred: Predicted voxels (any shape)
        target: Target voxels (same shape as pred)
        
    Returns:
        RMSE (lower is better)
    """
    mse = F.mse_loss(pred, target).item()
    return np.sqrt(mse)


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Dice Coefficient.
    
    Dice = 2 * |A  &  B| / (|A| + |B|)
    
    Args:
        pred: Predicted voxels
        target: Target voxels
        threshold: Threshold for binarization
        
    Returns:
        Dice coefficient in [0, 1] (higher is better)
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    intersection = (pred_flat * target_flat).sum().item()
    union = pred_flat.sum().item() + target_flat.sum().item()
    
    if union == 0:
        return 1.0
    
    return (2 * intersection) / union


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Intersection over Union (Jaccard Index).
    
    IoU = |A  &  B| / |A  |  B|
    
    Args:
        pred: Predicted voxels
        target: Target voxels
        threshold: Threshold for binarization
        
    Returns:
        IoU in [0, 1] (higher is better)
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    intersection = (pred_flat * target_flat).sum().item()
    union = pred_flat.sum().item() + target_flat.sum().item() - intersection
    
    if union == 0:
        return 1.0
    
    return intersection / union


def compute_precision_recall(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Compute Precision and Recall.
    
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    
    Args:
        pred: Predicted voxels
        target: Target voxels
        threshold: Threshold for binarization
        
    Returns:
        Tuple of (precision, recall)
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    return precision, recall


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        pred: Predicted voxels (B, 1, D, H, W) or (1, D, H, W) or (D, H, W)
        target: Target voxels (same shape as pred)
        threshold: Threshold for binarization
        
    Returns:
        Dictionary containing all metrics:
        - chamfer_distance: Chamfer Distance (lower is better)
        - psnr: Peak Signal-to-Noise Ratio in dB (higher is better)
        - ssim: Structural Similarity Index (higher is better)
        - rmse: Root Mean Square Error (lower is better)
        - dice: Dice Coefficient (higher is better)
        - iou: Intersection over Union (higher is better)
        - precision: Precision (higher is better)
        - recall: Recall (higher is better)
    """
    # Ensure tensors are on same device
    if pred.device != target.device:
        target = target.to(pred.device)
    
    # Compute all metrics
    chamfer = compute_chamfer_distance(pred, target, threshold=threshold)
    psnr = compute_psnr(pred, target)
    ssim = compute_ssim_3d(pred, target)
    rmse = compute_rmse(pred, target)
    dice = compute_dice(pred, target, threshold=threshold)
    iou = compute_iou(pred, target, threshold=threshold)
    precision, recall = compute_precision_recall(pred, target, threshold=threshold)
    
    return {
        'chamfer_distance': chamfer,
        'psnr': psnr,
        'ssim': ssim,
        'rmse': rmse,
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall
    }


class MetricsCalculator:
    """
    Class for accumulating and computing average metrics over batches.
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.metrics_sum = {
            'chamfer_distance': 0.0,
            'psnr': 0.0,
            'ssim': 0.0,
            'rmse': 0.0,
            'dice': 0.0,
            'iou': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
        self.count = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with a batch of predictions and targets.
        
        Args:
            pred: Predicted voxels (B, 1, D, H, W)
            target: Target voxels (B, 1, D, H, W)
        """
        batch_metrics = compute_all_metrics(pred, target, self.threshold)
        
        for key, value in batch_metrics.items():
            if not np.isinf(value):  # Handle inf PSNR
                self.metrics_sum[key] += value
        
        self.count += 1
    
    def compute(self) -> Dict[str, float]:
        """
        Compute average metrics.
        
        Returns:
            Dictionary of average metrics
        """
        if self.count == 0:
            return self.metrics_sum
        
        return {key: value / self.count for key, value in self.metrics_sum.items()}
    
    def __str__(self) -> str:
        """String representation of current metrics."""
        metrics = self.compute()
        lines = [
            f"Chamfer Distance: {metrics['chamfer_distance']:.6f}",
            f"PSNR: {metrics['psnr']:.2f} dB",
            f"SSIM: {metrics['ssim']:.4f}",
            f"RMSE: {metrics['rmse']:.6f}",
            f"Dice: {metrics['dice']:.4f}",
            f"IoU: {metrics['iou']:.4f}",
            f"Precision: {metrics['precision']:.4f}",
            f"Recall: {metrics['recall']:.4f}"
        ]
        return "\n".join(lines)


if __name__ == '__main__':
    # Test metrics
    print("Testing evaluation metrics...")
    
    # Create dummy predictions and targets
    torch.manual_seed(42)
    pred = torch.sigmoid(torch.randn(2, 1, 32, 32, 32))
    target = (torch.rand(2, 1, 32, 32, 32) > 0.7).float()
    
    # Compute all metrics
    metrics = compute_all_metrics(pred, target)
    
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
    
    # Test MetricsCalculator
    print("\n\nTesting MetricsCalculator...")
    calculator = MetricsCalculator()
    
    for _ in range(5):
        pred = torch.sigmoid(torch.randn(2, 1, 32, 32, 32))
        target = (torch.rand(2, 1, 32, 32, 32) > 0.7).float()
        calculator.update(pred, target)
    
    print("\nAverage Metrics:")
    print(calculator)
