"""
Dataset for TripoSR Mesh Fusion Training

Dataset Structure:
    /New_data/TripoSR_{energy_level}/00000{model_num}_gvxr_processed/view_{angle}/0/mesh.obj
    
    - 5 hand models: 000001 ~ 000005
    - 4 energy levels: e.g., TripoSR, TripoSR_0.08, TripoSR_0.1, etc.
    - Angles: range(-180, -134) + range(-45, 46) + range(135, 180)
    
Ground Truth:
    ./Bone_V1.stl, ./Bone_V2.stl, ..., ./Bone_V5.stl
    
Training Setup:
    - For each model, generate N random angle combinations
    - Total samples per energy level = 5 models x N combinations = 5N
    - Minimum angle separation enforced (e.g., 45 deg for 3 views, 30 deg for 6 views)
"""

import os
import random
import itertools
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
import warnings

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from .mesh_utils import (
        load_mesh, mesh_to_voxels, normalize_mesh,
        prepare_multi_view_input, NormalizationParams
    )
except ImportError:
    from mesh_utils import (
        load_mesh, mesh_to_voxels, normalize_mesh,
        prepare_multi_view_input, NormalizationParams
    )


# Valid angle ranges
VALID_ANGLES = list(range(-180, -134)) + list(range(-45, 46)) + list(range(135, 181))

# Model number to ground truth mapping
MODEL_TO_GT = {
    1: "Bone_V1.stl",
    2: "Bone_V2.stl",
    3: "Bone_V3.stl",
    4: "Bone_V4.stl",
    5: "Bone_V5.stl",
}


class AngleCombinationGenerator:
    """
    Generate random angle combinations with minimum separation constraint.
    
    Example:
        - 2 angles, min 45 deg: (-180, -135), (-180, -45), etc.
        - 3 angles, min 45 deg: (-180, -135, -45), (-180, -135, 0), etc.
        - 6 angles, min 30 deg: (-180, -150, -45, -15, 15, 135), etc.
    """
    
    def __init__(
        self,
        n_views: int,
        min_separation: int,
        available_angles: List[int] = None
    ):
        """
        Args:
            n_views: Number of angles to select
            min_separation: Minimum angular separation between views
            available_angles: List of available angles (default: VALID_ANGLES)
        """
        self.n_views = n_views
        self.min_separation = min_separation
        self.available_angles = available_angles or VALID_ANGLES
    
    def _angular_distance(self, a1: int, a2: int) -> int:
        """Compute angular distance considering wrap-around."""
        diff = abs(a1 - a2)
        return min(diff, 360 - diff)
    
    def _is_valid_combination(self, angles: List[int]) -> bool:
        """Check if all angles have sufficient separation."""
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                if self._angular_distance(angles[i], angles[j]) < self.min_separation:
                    return False
        return True
    
    def generate_single(self, seed: int = None) -> List[int]:
        """Generate a single valid angle combination."""
        if seed is not None:
            random.seed(seed)
        
        # Greedy random selection
        max_attempts = 1000
        for _ in range(max_attempts):
            candidates = self.available_angles.copy()
            random.shuffle(candidates)
            
            selected = []
            for angle in candidates:
                if len(selected) >= self.n_views:
                    break
                if all(self._angular_distance(angle, s) >= self.min_separation for s in selected):
                    selected.append(angle)
            
            if len(selected) == self.n_views:
                return sorted(selected)
        
        raise ValueError(f"Could not generate valid combination with {self.n_views} views "
                        f"and {self.min_separation} deg separation after {max_attempts} attempts")
    
    def generate_multiple(self, n_combinations: int, seed: int = 42) -> List[List[int]]:
        """
        Generate N unique angle combinations.
        
        Args:
            n_combinations: Number of combinations to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of angle combinations
        """
        random.seed(seed)
        np.random.seed(seed)
        
        combinations = set()
        attempts = 0
        max_attempts = n_combinations * 100
        
        while len(combinations) < n_combinations and attempts < max_attempts:
            combo = tuple(self.generate_single())
            combinations.add(combo)
            attempts += 1
        
        if len(combinations) < n_combinations:
            warnings.warn(f"Could only generate {len(combinations)} unique combinations "
                         f"(requested {n_combinations})")
        
        return [list(c) for c in combinations]
    
    def get_all_valid_combinations(self, max_count: int = None) -> List[List[int]]:
        """
        Get all valid combinations (for small n_views).
        Warning: Can be very large for many views!
        """
        all_combos = list(itertools.combinations(self.available_angles, self.n_views))
        valid = [list(c) for c in all_combos if self._is_valid_combination(list(c))]
        
        if max_count and len(valid) > max_count:
            random.shuffle(valid)
            valid = valid[:max_count]
        
        return valid


class TripoSRDataset(Dataset):
    """
    Dataset for TripoSR mesh fusion training.
    
    Structure:
        base_path/TripoSR_{energy}/00000{model}_gvxr_processed/view_{angle}/0/mesh.obj
        gt_path/Bone_V{model}.stl
    
    Args:
        base_path: Base path containing TripoSR folders
        gt_path: Path to ground truth STL files
        energy_level: Energy level string (e.g., "", "0.08", "0.1")
        n_views: Number of viewing angles per sample
        n_combinations: Number of angle combinations per model (N)
        min_angle_separation: Minimum degrees between angles
        resolution: Voxel grid resolution
        seed: Random seed
        
    Total samples = 5 models x n_combinations
    """
    
    def __init__(
        self,
        base_path: Union[str, Path],
        gt_path: Union[str, Path],
        energy_level: str = "",  # "" for TripoSR, "0.08" for TripoSR_0.08, etc.
        n_views: int = 3,
        n_combinations: int = 100,  # N combinations per model
        min_angle_separation: int = 45,
        resolution: int = 64,
        cache_voxels: bool = True,
        seed: int = 42
    ):
        self.base_path = Path(base_path)
        self.gt_path = Path(gt_path)
        self.n_views = n_views
        self.n_combinations = n_combinations
        self.min_angle_separation = min_angle_separation
        self.resolution = resolution
        self.seed = seed
        
        # Determine energy folder name
        if energy_level == "" or energy_level is None:
            self.energy_folder = "TripoSR"
        else:
            self.energy_folder = f"TripoSR_{energy_level}"
        
        self.energy_path = self.base_path / self.energy_folder
        
        if not self.energy_path.exists():
            raise FileNotFoundError(f"Energy folder not found: {self.energy_path}")
        
        # Setup cache
        self.cache_voxels = cache_voxels
        if cache_voxels:
            self.cache_dir = self.base_path / '.cache' / self.energy_folder / f'r{resolution}_v{n_views}_n{n_combinations}'
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Discover models and generate angle combinations
        self.samples = self._build_dataset()
        
        print(f"Dataset [{self.energy_folder}]:")
        print(f"  Models: 5")
        print(f"  Combinations per model: {n_combinations}")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Views per sample: {n_views}")
        print(f"  Min angle separation: {min_angle_separation} deg")
    
    def _build_dataset(self) -> List[Dict]:
        """Build dataset with N angle combinations per model."""
        samples = []
        
        # Generate angle combinations
        generator = AngleCombinationGenerator(
            n_views=self.n_views,
            min_separation=self.min_angle_separation,
            available_angles=VALID_ANGLES
        )
        
        angle_combinations = generator.generate_multiple(
            n_combinations=self.n_combinations,
            seed=self.seed
        )
        
        print(f"  Generated {len(angle_combinations)} angle combinations")
        
        # For each model (1-5)
        for model_num in range(1, 6):
            model_id = f"00000{model_num}_gvxr_processed"
            model_path = self.energy_path / model_id
            
            if not model_path.exists():
                warnings.warn(f"Model folder not found: {model_path}")
                continue
            
            # Get ground truth path
            gt_filename = MODEL_TO_GT.get(model_num)
            gt_file = self.gt_path / gt_filename
            
            if not gt_file.exists():
                warnings.warn(f"Ground truth not found: {gt_file}")
                continue
            
            # Find available angles for this model
            available_angles = self._find_available_angles(model_path)
            
            # Create sample for each angle combination
            for combo_idx, angles in enumerate(angle_combinations):
                # Check if all angles are available for this model
                if all(a in available_angles for a in angles):
                    samples.append({
                        'model_num': model_num,
                        'model_id': model_id,
                        'model_path': model_path,
                        'gt_path': gt_file,
                        'angles': angles,
                        'combo_idx': combo_idx
                    })
        
        return samples
    
    def _find_available_angles(self, model_path: Path) -> List[int]:
        """Find all available angles for a model."""
        angles = []
        for view_dir in model_path.iterdir():
            if not view_dir.is_dir() or not view_dir.name.startswith('view_'):
                continue
            try:
                # Extract angle from folder name (handles view_+45, view_-45, view_0, etc.)
                angle_str = view_dir.name.replace('view_', '')
                # Remove + sign if present (e.g., "+45" -> "45")
                angle_str = angle_str.lstrip('+')
                angle = int(angle_str)
                mesh_path = view_dir / '0' / 'mesh.obj'
                if mesh_path.exists():
                    angles.append(angle)
            except ValueError:
                continue
        return sorted(angles)
    
    def _get_view_folder_name(self, angle: int) -> str:
        """Get the folder name for a given angle (handles +/- sign)."""
        if angle >= 0:
            return f"view_+{angle}"
        else:
            return f"view_{angle}"  # Negative angles already have minus sign
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        sample = self.samples[idx]
        
        # Cache key
        angles_str = '_'.join(map(str, sample['angles']))
        cache_key = f"m{sample['model_num']}_c{sample['combo_idx']}_{angles_str}"
        
        # Try cache
        if self.cache_voxels:
            cache_input = self.cache_dir / f'{cache_key}_input.npy'
            cache_target = self.cache_dir / f'{cache_key}_target.npy'
            cache_params = self.cache_dir / f'{cache_key}_params.npz'
            
            if cache_input.exists() and cache_target.exists() and cache_params.exists():
                inputs = torch.from_numpy(np.load(cache_input).astype(np.float32))
                target = torch.from_numpy(np.load(cache_target).astype(np.float32))
                if target.dim() == 3:
                    target = target.unsqueeze(0)
                
                # Load normalization params
                params_data = np.load(cache_params)
                gt_center = params_data['center'].tolist()
                gt_scale = float(params_data['scale'])
                gt_bounds = params_data['bounds'].tolist()
                
                return inputs, target, {
                    'model_num': sample['model_num'],
                    'angles': sample['angles'],
                    'gt_path': str(sample['gt_path']),
                    'idx': idx,
                    'gt_center': gt_center,
                    'gt_scale': gt_scale,
                    'gt_bounds': gt_bounds
                }
        
        # Load input meshes (handle +/- sign in folder names)
        mesh_paths = [
            sample['model_path'] / self._get_view_folder_name(angle) / '0' / 'mesh.obj'
            for angle in sample['angles']
        ]
        inputs = prepare_multi_view_input(
            mesh_paths,
            resolution=self.resolution,
            align=True,
            fill_interior=True
        )
        
        # Load ground truth and get normalization parameters
        gt_mesh = load_mesh(sample['gt_path'])
        
        # Store original bounds BEFORE normalization
        gt_bounds_original = gt_mesh.bounds.tolist()  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        gt_center_original = gt_mesh.centroid.copy()
        
        # Normalize
        gt_mesh_norm, norm_params = normalize_mesh(gt_mesh, return_params=True)
        target = mesh_to_voxels(gt_mesh_norm, resolution=self.resolution, fill_interior=True)
        
        # Cache
        if self.cache_voxels:
            np.save(cache_input, inputs)
            np.save(cache_target, target)
            np.savez(cache_params, 
                     center=norm_params.center, 
                     scale=norm_params.scale,
                     bounds=np.array(gt_bounds_original))
        
        inputs = torch.from_numpy(inputs.astype(np.float32))
        target = torch.from_numpy(target[np.newaxis].astype(np.float32))
        
        return inputs, target, {
            'model_num': sample['model_num'],
            'angles': sample['angles'],
            'gt_path': str(sample['gt_path']),
            'idx': idx,
            'gt_center': gt_center_original.tolist(),
            'gt_scale': norm_params.scale,
            'gt_bounds': gt_bounds_original
        }


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    inputs = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    metadata = [b[2] for b in batch]
    return inputs, targets, metadata


def create_dataloaders(
    base_path: Union[str, Path],
    gt_path: Union[str, Path],
    energy_level: str = "",
    n_views: int = 3,
    n_combinations: int = 100,
    min_angle_separation: int = 45,
    resolution: int = 64,
    batch_size: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Total samples = 5 models x n_combinations
    """
    dataset = TripoSRDataset(
        base_path=base_path,
        gt_path=gt_path,
        energy_level=energy_level,
        n_views=n_views,
        n_combinations=n_combinations,
        min_angle_separation=min_angle_separation,
        resolution=resolution,
        seed=seed
    )
    
    # Split
    n = len(dataset)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_test = n - n_train - n_val
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Note: On Windows, use num_workers=0 if you get pickling errors
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, 
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    
    print(f"\nDataloader split:")
    print(f"  Train: {n_train} samples ({n_train // batch_size} batches)")
    print(f"  Val: {n_val} samples")
    print(f"  Test: {n_test} samples")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test angle combination generator
    print("Testing AngleCombinationGenerator...")
    
    # 3 views, 45 deg separation
    gen3 = AngleCombinationGenerator(n_views=3, min_separation=45)
    combos3 = gen3.generate_multiple(10, seed=42)
    print(f"\n3 views, 45 deg separation (10 combinations):")
    for c in combos3[:5]:
        print(f"  {c}")
    
    # 6 views, 30 deg separation
    gen6 = AngleCombinationGenerator(n_views=6, min_separation=30)
    combos6 = gen6.generate_multiple(10, seed=42)
    print(f"\n6 views, 30 deg separation (10 combinations):")
    for c in combos6[:5]:
        print(f"  {c}")
    
    # 2 views, 45 deg separation
    gen2 = AngleCombinationGenerator(n_views=2, min_separation=45)
    combos2 = gen2.generate_multiple(10, seed=42)
    print(f"\n2 views, 45 deg separation (10 combinations):")
    for c in combos2[:5]:
        print(f"  {c}")
    
    print("\n--- Dataset structure ---")
    print("Expected folder structure:")
    print("  base_path/TripoSR_{energy}/00000{1-5}_gvxr_processed/view_{angle}/0/mesh.obj")
    print("  gt_path/Bone_V{1-5}.stl")
    print("\nTotal samples = 5 models x N combinations")
