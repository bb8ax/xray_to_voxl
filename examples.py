#!/usr/bin/env python
"""
Example Usage Script for TripoSR GVXR Dataset Training

This script demonstrates various ways to configure and run training
with different numbers of views, angle strategies, and energy folders.

Your data structure:
    /TripoSR/000001_gvxr_processed/view_{angle}/0/mesh.obj
    /TripoSR_0.08/000001_gvxr_processed/view_{angle}/0/mesh.obj
    
Valid angles: -180 ~ -135, -45 ~ 45, 135 ~ 180
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset_gvxr import AngleSelector, TripoSRGVXRDatasetFixedAngles


def example_angle_selection():
    """Demonstrate angle selection strategies."""
    print("\n" + "="*60)
    print("ANGLE SELECTION EXAMPLES")
    print("="*60)
    
    selector = AngleSelector(min_separation=45)
    
    print("\nValid angle ranges:", AngleSelector.VALID_RANGES)
    print(f"Total valid angles: {len(selector.valid_angles)}")
    print(f"Sample valid angles: {selector.valid_angles[::20]}")  # Every 20th
    
    print("\n--- Strategy: 'random' (different each time, respects min separation) ---")
    for n_views in [2, 3, 4, 5, 6]:
        angles = selector.select_angles(n_views, strategy='random', seed=42)
        print(f"  n_views={n_views}: {angles}")
    
    print("\n--- Strategy: 'uniform' (evenly distributed) ---")
    for n_views in [2, 3, 4, 5, 6]:
        angles = selector.select_angles(n_views, strategy='uniform', seed=42)
        print(f"  n_views={n_views}: {angles}")
    
    print("\n--- Strategy: 'fixed' (predefined good combinations) ---")
    for n_views in [2, 3, 4, 5, 6]:
        angles = selector.select_angles(n_views, strategy='fixed', seed=42)
        print(f"  n_views={n_views}: {angles}")
    
    print("\n--- Valid 3-view combinations (sample) ---")
    combos = selector.get_all_valid_combinations(3, max_combinations=10)
    for combo in combos[:5]:
        print(f"  {combo}")
    print(f"  ... total valid combinations: {len(selector.get_all_valid_combinations(3, max_combinations=10000))}")


def print_training_commands():
    """Print example training commands."""
    print("\n" + "="*60)
    print("EXAMPLE TRAINING COMMANDS")
    print("="*60)
    
    commands = [
        # Basic 3-view training
        """
# Basic training with 3 views
python train_gvxr.py \\
    --base_path /path/to/data \\
    --ground_truth_path /path/to/ground_truth \\
    --n_views 3 \\
    --epochs 100 \\
    --batch_size 4
""",
        # 5-view training with specific energy
        """
# Training with 5 views, single energy folder
python train_gvxr.py \\
    --base_path /path/to/data \\
    --ground_truth_path /path/to/ground_truth \\
    --n_views 5 \\
    --energy_folders "TripoSR" \\
    --angle_strategy uniform \\
    --epochs 100
""",
        # Multiple energy folders
        """
# Training with multiple energy folders
python train_gvxr.py \\
    --base_path /path/to/data \\
    --ground_truth_path /path/to/ground_truth \\
    --n_views 3 \\
    --energy_folders "TripoSR,TripoSR_0.08" \\
    --epochs 100
""",
        # High resolution with attention model
        """
# High resolution training with attention model
python train_gvxr.py \\
    --base_path /path/to/data \\
    --ground_truth_path /path/to/ground_truth \\
    --n_views 4 \\
    --resolution 128 \\
    --model_type attention \\
    --base_features 64 \\
    --batch_size 2 \\
    --epochs 150
""",
        # Different angle strategies
        """
# Compare different angle selection strategies:

# Random angles (different per sample, ensures variety)
python train_gvxr.py --n_views 3 --angle_strategy random ...

# Uniform angles (evenly distributed across valid ranges)  
python train_gvxr.py --n_views 3 --angle_strategy uniform ...

# Fixed angles (predefined good combinations)
python train_gvxr.py --n_views 3 --angle_strategy fixed ...
""",
    ]
    
    for cmd in commands:
        print(cmd)


def print_data_structure():
    """Print expected data structure."""
    print("\n" + "="*60)
    print("EXPECTED DATA STRUCTURE")
    print("="*60)
    
    structure = """
Your data should be organized as:

base_path/
|--- TripoSR/                              # Standard energy
|   |--- 000001_gvxr_processed/
|   |   |--- view_-180/
|   |   |   \--- 0/
|   |   |       \--- mesh.obj              # TripoSR output
|   |   |--- view_-135/
|   |   |   \--- 0/
|   |   |       \--- mesh.obj
|   |   |--- view_-45/
|   |   |   \--- 0/
|   |   |       \--- mesh.obj
|   |   |--- view_0/
|   |   |   \--- 0/
|   |   |       \--- mesh.obj
|   |   |--- view_45/
|   |   |   \--- 0/
|   |   |       \--- mesh.obj
|   |   |--- view_135/
|   |   |   \--- 0/
|   |   |       \--- mesh.obj
|   |   \--- view_180/
|   |       \--- 0/
|   |           \--- mesh.obj
|   |--- 000002_gvxr_processed/
|   |   \--- ...
|   \--- ...
|--- TripoSR_0.08/                         # Different X-ray energy
|   |--- 000001_gvxr_processed/
|   |   \--- ...
|   \--- ...
\--- TripoSR_0.1/                          # Another energy
    \--- ...

ground_truth_path/
|--- 000001_gvxr_processed/
|   \--- bone.obj                          # Ground truth bone model
|--- 000002_gvxr_processed/
|   \--- bone.obj
\--- ...

Valid angle ranges for views:
  - -180 to -135 degrees
  - -45 to 45 degrees  
  - 135 to 180 degrees

Note: Avoid selecting adjacent angles (e.g., -180, -179, -178).
      Minimum separation of 45 degrees is recommended.
"""
    print(structure)


def python_api_example():
    """Show Python API usage example."""
    print("\n" + "="*60)
    print("PYTHON API EXAMPLE")
    print("="*60)
    
    code = '''
from data.dataset_gvxr import (
    TripoSRGVXRDatasetFixedAngles,
    create_gvxr_data_loaders,
    AngleSelector
)
from models.unet3d import get_model
import torch

# 1. Create dataset directly
dataset = TripoSRGVXRDatasetFixedAngles(
    base_path='/path/to/data',
    ground_truth_path='/path/to/ground_truth',
    n_views=3,                      # Number of input views
    resolution=64,                  # Voxel resolution
    energy_folders=['TripoSR'],     # Which energy folders to use
    angle_strategy='random',        # How to select angles
    min_angle_separation=45,        # Minimum degrees between views
    seed=42
)

print(f"Dataset size: {len(dataset)}")

# Get a sample
inputs, target, metadata = dataset[0]
print(f"Input shape: {inputs.shape}")      # (n_views, D, H, W)
print(f"Target shape: {target.shape}")     # (1, D, H, W)
print(f"Selected angles: {metadata['selected_angles']}")

# 2. Create data loaders
train_loader, val_loader, test_loader = create_gvxr_data_loaders(
    base_path='/path/to/data',
    ground_truth_path='/path/to/ground_truth',
    n_views=4,
    batch_size=4,
    resolution=64,
    energy_folders=['TripoSR', 'TripoSR_0.08'],  # Multiple energies
    angle_strategy='uniform',
    min_angle_separation=45,
    train_split=0.8,
    val_split=0.1,
    seed=42
)

# 3. Create model (input channels = n_views)
model = get_model(
    model_type='attention',
    in_channels=4,      # Must match n_views!
    out_channels=1,
    base_features=32,
    depth=4
)

# 4. Training loop example
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for inputs, targets, metadata in train_loader:
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    outputs = model(inputs)
    # ... compute loss, backprop, etc.
    
    # Access metadata
    for m in metadata:
        print(f"Sample: {m['sample_id']}, Angles: {m['selected_angles']}")
'''
    print(code)


def main():
    """Run all examples."""
    example_angle_selection()
    print_data_structure()
    print_training_commands()
    python_api_example()
    
    print("\n" + "="*60)
    print("QUICK START")
    print("="*60)
    print("""
1. Organize your data as shown above

2. Run training:
   python train_gvxr.py \\
       --base_path /path/to/TripoSR_data \\
       --ground_truth_path /path/to/ground_truth \\
       --n_views 3 \\
       --epochs 100

3. Monitor with TensorBoard:
   tensorboard --logdir ./outputs

4. Run inference:
   python inference.py \\
       --checkpoint ./outputs/run_xxx/best_model.pt \\
       --input_dir /path/to/sample \\
       --output_dir ./results
""")


if __name__ == '__main__':
    main()
