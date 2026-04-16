"""
Training Script for 3D U-Net TripoSR Mesh Fusion with GVXR Dataset

This script handles:
1. Model training with configurable hyperparameters
2. Configurable number of input angles
3. Smart angle selection with minimum separation
4. Support for multiple energy folders

Usage:
    python train_gvxr.py \
        --base_path /path/to/TripoSR_data \
        --ground_truth_path /path/to/ground_truth \
        --n_views 3 \
        --epochs 100 \
        --batch_size 4
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.unet3d import UNet3D, UNet3DMultiScale, get_model
from data.dataset_gvxr import (
    TripoSRGVXRDataset, 
    TripoSRGVXRDatasetFixedAngles,
    create_gvxr_data_loaders,
    AngleSelector,
    custom_collate_fn
)
from losses import get_loss_function, MultiScaleLoss


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self, 
        patience: int = 10, 
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class MetricTracker:
    """Track and compute metrics during training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.values = {}
        self.counts = {}
    
    def update(self, name: str, value: float, count: int = 1):
        if name not in self.values:
            self.values[name] = 0.0
            self.counts[name] = 0
        self.values[name] += value * count
        self.counts[name] += count
    
    def get(self, name: str) -> float:
        if name not in self.values:
            return 0.0
        return self.values[name] / max(self.counts[name], 1)
    
    def get_all(self) -> Dict[str, float]:
        return {name: self.get(name) for name in self.values}


def compute_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
    
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_deep_supervision: bool = False
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    metrics = MetricTracker()
    
    for batch_idx, (inputs, targets, metadata) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        if use_deep_supervision:
            outputs, deep_outputs = model(inputs, return_deep=True)
            loss = criterion(outputs, deep_outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        with torch.no_grad():
            batch_metrics = compute_metrics(outputs, targets)
        
        batch_size = inputs.size(0)
        metrics.update('loss', loss.item(), batch_size)
        for name, value in batch_metrics.items():
            metrics.update(name, value, batch_size)
        
        if batch_idx % 10 == 0:
            # Show sample info
            sample_ids = [m['sample_id'] for m in metadata[:2]]
            angles = [m['selected_angles'] for m in metadata[:2]]
            print(f"  Batch {batch_idx}/{len(train_loader)} - "
                  f"Loss: {loss.item():.4f}, Dice: {batch_metrics['dice']:.4f}")
            print(f"    Samples: {sample_ids[:2]}, Angles: {angles[:2]}")
    
    return metrics.get_all()


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    metrics = MetricTracker()
    
    with torch.no_grad():
        for inputs, targets, metadata in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            batch_metrics = compute_metrics(outputs, targets)
            
            batch_size = inputs.size(0)
            metrics.update('loss', loss.item(), batch_size)
            for name, value in batch_metrics.items():
                metrics.update(name, value, batch_size)
    
    return metrics.get_all()


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    filepath: Path,
    config: Dict
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Any = None
) -> Tuple[int, Dict[str, float]]:
    """Load training checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics']


def train(config: Dict):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config['output_dir']) / f"run_{timestamp}_n{config['n_views']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Parse energy folders
    energy_folders = None
    if config.get('energy_folders'):
        energy_folders = config['energy_folders'].split(',')
    
    # Create data loaders
    print("Creating data loaders...")
    print(f"  Base path: {config['base_path']}")
    print(f"  Ground truth path: {config['ground_truth_path']}")
    print(f"  Number of views: {config['n_views']}")
    print(f"  Angle strategy: {config['angle_strategy']}")
    print(f"  Min angle separation: {config['min_angle_separation']} deg")
    
    train_loader, val_loader, test_loader = create_gvxr_data_loaders(
        base_path=config['base_path'],
        ground_truth_path=config['ground_truth_path'],
        n_views=config['n_views'],
        batch_size=config['batch_size'],
        resolution=config['resolution'],
        energy_folders=energy_folders,
        angle_strategy=config['angle_strategy'],
        min_angle_separation=config['min_angle_separation'],
        train_split=config.get('train_split', 0.8),
        val_split=config.get('val_split', 0.1),
        num_workers=config.get('num_workers', 4),
        seed=config.get('seed', 42)
    )
    
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")
    
    # Create model
    print("Creating model...")
    model = get_model(
        model_type=config.get('model_type', 'attention'),
        in_channels=config['n_views'],  # Number of input views
        out_channels=1,
        base_features=config.get('base_features', 32),
        depth=config.get('depth', 4),
        use_residual=config.get('use_residual', True),
        dropout_rate=config.get('dropout_rate', 0.1)
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input channels (views): {config['n_views']}")
    
    # Create loss function
    use_deep_supervision = config.get('model_type') == 'multiscale'
    base_loss = get_loss_function(config.get('loss_type', 'dice_bce'))
    
    if use_deep_supervision:
        criterion = MultiScaleLoss(base_loss, scales=config.get('depth', 4))
    else:
        criterion = base_loss
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.get('scheduler_T0', 10),
        T_mult=2,
        eta_min=config.get('min_lr', 1e-6)
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get('patience', 15),
        min_delta=1e-4,
        mode='min'
    )
    
    # TensorBoard logging
    writer = SummaryWriter(output_dir / 'logs')
    
    # Log hyperparameters
    writer.add_text('config', json.dumps(config, indent=2))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config.get('resume_checkpoint'):
        print(f"Resuming from checkpoint: {config['resume_checkpoint']}")
        start_epoch, _ = load_checkpoint(
            config['resume_checkpoint'], model, optimizer, scheduler
        )
        start_epoch += 1
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    for epoch in range(start_epoch, config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 40)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_deep_supervision=use_deep_supervision
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        print(f"Learning rate: {current_lr:.2e}")
        
        # TensorBoard logging
        for name, value in train_metrics.items():
            writer.add_scalar(f'train/{name}', value, epoch)
        for name, value in val_metrics.items():
            writer.add_scalar(f'val/{name}', value, epoch)
        writer.add_scalar('learning_rate', current_lr, epoch)
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                output_dir / 'best_model.pt', config
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % config.get('checkpoint_interval', 10) == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                output_dir / f'checkpoint_epoch_{epoch + 1}.pt', config
            )
        
        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    # Final evaluation
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    load_checkpoint(output_dir / 'best_model.pt', model)
    test_metrics = validate_epoch(model, test_loader, criterion, device)
    
    print("\nTest Results:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Save test results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    writer.close()
    print(f"\nTraining complete. Results saved to: {output_dir}")
    
    return output_dir, test_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train 3D U-Net for TripoSR Mesh Fusion with GVXR Dataset'
    )
    
    # Data arguments
    parser.add_argument('--base_path', type=str, required=True,
                        help='Base path containing TripoSR folders')
    parser.add_argument('--ground_truth_path', type=str, required=True,
                        help='Path to ground truth meshes')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')
    
    # View/Angle arguments
    parser.add_argument('--n_views', type=int, default=3,
                        help='Number of input views (angles)')
    parser.add_argument('--angle_strategy', type=str, default='random',
                        choices=['random', 'uniform', 'fixed'],
                        help='Strategy for selecting angles')
    parser.add_argument('--min_angle_separation', type=int, default=45,
                        help='Minimum angle separation between views (degrees)')
    parser.add_argument('--energy_folders', type=str, default=None,
                        help='Comma-separated list of energy folders (e.g., "TripoSR,TripoSR_0.08")')
    
    # Resolution
    parser.add_argument('--resolution', type=int, default=64,
                        help='Voxel grid resolution')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='attention',
                        choices=['standard', 'attention', 'multiscale'],
                        help='Model architecture type')
    parser.add_argument('--base_features', type=int, default=32,
                        help='Base feature count')
    parser.add_argument('--depth', type=int, default=4,
                        help='Network depth')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--loss_type', type=str, default='dice_bce',
                        choices=['dice', 'bce', 'dice_bce', 'focal', 'tversky'],
                        help='Loss function type')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    config = vars(args)
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Print configuration
    print("\n" + "="*60)
    print("Configuration:")
    print("="*60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Train
    train(config)


if __name__ == '__main__':
    main()
