"""
Dataset Classes for TripoSR Mesh Fusion Training

Updated to support:
1. Configurable number of input angles (n views)
2. Smart angle selection with minimum separation
3. Specific folder structure: /TripoSR{_energy}/XXXXXX_gvxr_processed/view_{angle}/0/mesh.obj

Valid angle ranges: -180 ~ -135, -45 ~ 45, 135 ~ 180
Minimum angle separation: 45 degrees
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Callable, Set
import warnings
import itertools

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from .mesh_utils import (
        load_obj_mesh, mesh_to_voxels, prepare_multi_view_input,
        normalize_mesh, MeshProcessor
    )
    MESH_UTILS_AVAILABLE = True
except ImportError:
    try:
        from mesh_utils import (
            load_obj_mesh, mesh_to_voxels, prepare_multi_view_input,
            normalize_mesh, MeshProcessor
        )
        MESH_UTILS_AVAILABLE = True
    except ImportError:
        MESH_UTILS_AVAILABLE = False
        warnings.warn("mesh_utils not available")


class AngleSelector:
    """
    Handles smart selection of viewing angles for training.
    
    Valid angle ranges: -180 ~ -135, -45 ~ 45, 135 ~ 180
    Ensures minimum separation between selected angles.
    """
    
    # Valid angle ranges (in degrees)
    VALID_RANGES = [
        (-180, -135),
        (-45, 45),
        (135, 180)
    ]
    
    def __init__(
        self,
        min_separation: int = 45,
        available_angles: Optional[List[int]] = None
    ):
        """
        Args:
            min_separation: Minimum angular separation between selected views
            available_angles: List of available angles (if None, uses all valid angles)
        """
        self.min_separation = min_separation
        
        if available_angles is not None:
            self.available_angles = sorted(available_angles)
        else:
            # Generate all valid integer angles
            self.available_angles = self._get_all_valid_angles()
        
        # Filter to only valid ranges
        self.valid_angles = [a for a in self.available_angles if self._is_valid_angle(a)]
    
    def _get_all_valid_angles(self) -> List[int]:
        """Generate all valid integer angles from valid ranges."""
        angles = []
        for start, end in self.VALID_RANGES:
            angles.extend(range(start, end + 1))
        return sorted(angles)
    
    def _is_valid_angle(self, angle: int) -> bool:
        """Check if angle is within valid ranges."""
        for start, end in self.VALID_RANGES:
            if start <= angle <= end:
                return True
        return False
    
    def _angular_distance(self, a1: int, a2: int) -> int:
        """
        Compute angular distance considering wrap-around.
        E.g., distance between -180 and 180 is 0 (same direction).
        """
        diff = abs(a1 - a2)
        return min(diff, 360 - diff)
    
    def _is_valid_combination(self, angles: List[int]) -> bool:
        """Check if all angles in combination have sufficient separation."""
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                if self._angular_distance(angles[i], angles[j]) < self.min_separation:
                    return False
        return True
    
    def select_angles(
        self,
        n_views: int,
        strategy: str = 'random',
        seed: Optional[int] = None
    ) -> List[int]:
        """
        Select n angles with minimum separation.
        
        Args:
            n_views: Number of angles to select
            strategy: Selection strategy ('random', 'uniform', 'fixed')
            seed: Random seed for reproducibility
            
        Returns:
            List of selected angles
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        if n_views > len(self.valid_angles):
            raise ValueError(f"Cannot select {n_views} views from {len(self.valid_angles)} available angles")
        
        if strategy == 'uniform':
            return self._select_uniform(n_views)
        elif strategy == 'random':
            return self._select_random(n_views)
        elif strategy == 'fixed':
            return self._select_fixed(n_views)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _select_uniform(self, n_views: int) -> List[int]:
        """Select angles uniformly distributed across valid ranges."""
        # Spread across the full 360 degrees, then filter to valid
        target_angles = []
        step = 360 / n_views
        
        for i in range(n_views):
            target = int(-180 + i * step)
            # Find closest valid angle
            closest = min(self.valid_angles, key=lambda x: self._angular_distance(x, target))
            if closest not in target_angles:
                target_angles.append(closest)
        
        # If we didn't get enough unique angles, fill with random valid ones
        while len(target_angles) < n_views:
            remaining = [a for a in self.valid_angles if a not in target_angles]
            if not remaining:
                break
            target_angles.append(random.choice(remaining))
        
        return sorted(target_angles)
    
    def _select_random(self, n_views: int, max_attempts: int = 1000) -> List[int]:
        """Select random angles with minimum separation."""
        for _ in range(max_attempts):
            selected = random.sample(self.valid_angles, n_views)
            if self._is_valid_combination(selected):
                return sorted(selected)
        
        # Fallback: use greedy selection
        return self._select_greedy(n_views)
    
    def _select_greedy(self, n_views: int) -> List[int]:
        """Greedy selection ensuring minimum separation."""
        selected = []
        candidates = self.valid_angles.copy()
        random.shuffle(candidates)
        
        for angle in candidates:
            if len(selected) >= n_views:
                break
            
            # Check if this angle is far enough from all selected
            valid = True
            for selected_angle in selected:
                if self._angular_distance(angle, selected_angle) < self.min_separation:
                    valid = False
                    break
            
            if valid:
                selected.append(angle)
        
        if len(selected) < n_views:
            warnings.warn(f"Could only select {len(selected)} angles with min_separation={self.min_separation}")
        
        return sorted(selected)
    
    def _select_fixed(self, n_views: int) -> List[int]:
        """Select fixed, well-distributed angles."""
        # Predefined good combinations for common n_views
        fixed_combinations = {
            2: [0, 135],
            3: [-135, 0, 135],
            4: [-180, -45, 45, 135],
            5: [-180, -135, 0, 45, 135],
            6: [-180, -135, -45, 45, 135, 180],
        }
        
        if n_views in fixed_combinations:
            # Filter to available angles
            combo = fixed_combinations[n_views]
            result = [a for a in combo if a in self.valid_angles]
            if len(result) == n_views:
                return result
        
        # Fallback to uniform
        return self._select_uniform(n_views)
    
    def get_all_valid_combinations(self, n_views: int, max_combinations: int = 10000) -> List[List[int]]:
        """
        Get all valid combinations of n angles.
        
        Useful for exhaustive training or validation.
        """
        all_combos = list(itertools.combinations(self.valid_angles, n_views))
        valid_combos = [list(c) for c in all_combos if self._is_valid_combination(list(c))]
        
        if len(valid_combos) > max_combinations:
            return random.sample(valid_combos, max_combinations)
        
        return valid_combos


class TripoSRGVXRDataset(Dataset):
    """
    Dataset for TripoSR mesh fusion with gvxr ground truth.
    
    Folder structure:
        base_path/TripoSR{_energy}/XXXXXX_gvxr_processed/view_{angle}/0/mesh.obj
        
    Examples:
        /TripoSR/000001_gvxr_processed/view_0/0/mesh.obj
        /TripoSR_0.08/000001_gvxr_processed/view_-45/0/mesh.obj
    
    Ground truth should be in:
        ground_truth_path/XXXXXX_gvxr_processed/bone.obj (or similar)
    
    Args:
        base_path: Base path containing TripoSR folders
        ground_truth_path: Path to ground truth meshes
        n_views: Number of views to use per sample
        resolution: Voxel grid resolution
        energy_folders: List of energy folders to use (e.g., ['TripoSR', 'TripoSR_0.08'])
        angle_strategy: How to select angles ('random', 'uniform', 'fixed')
        min_angle_separation: Minimum degrees between selected angles
        transform: Optional transform for inputs
        target_transform: Optional transform for targets
        cache_voxels: Whether to cache voxelized data
        seed: Random seed for angle selection
    """
    
    def __init__(
        self,
        base_path: Union[str, Path],
        ground_truth_path: Union[str, Path],
        n_views: int = 3,
        resolution: int = 64,
        energy_folders: Optional[List[str]] = None,
        angle_strategy: str = 'random',
        min_angle_separation: int = 45,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_voxels: bool = True,
        fill_interior: bool = True,
        seed: int = 42
    ):
        self.base_path = Path(base_path)
        self.ground_truth_path = Path(ground_truth_path)
        self.n_views = n_views
        self.resolution = resolution
        self.angle_strategy = angle_strategy
        self.min_angle_separation = min_angle_separation
        self.transform = transform
        self.target_transform = target_transform
        self.cache_voxels = cache_voxels
        self.fill_interior = fill_interior
        self.seed = seed
        
        # Find energy folders
        if energy_folders is None:
            energy_folders = self._find_energy_folders()
        self.energy_folders = energy_folders
        
        # Create cache directory
        if cache_voxels:
            self.cache_dir = self.base_path / '.cache' / f'res{resolution}_n{n_views}'
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Discover all samples
        self.samples = self._discover_samples()
        
        # Initialize angle selector
        self.angle_selector = AngleSelector(min_separation=min_angle_separation)
        
        print(f"Found {len(self.samples)} samples across {len(self.energy_folders)} energy folders")
        print(f"Energy folders: {self.energy_folders}")
        print(f"Using {n_views} views with {angle_strategy} angle selection")
    
    def _find_energy_folders(self) -> List[str]:
        """Find all TripoSR energy folders."""
        folders = []
        for item in self.base_path.iterdir():
            if item.is_dir() and item.name.startswith('TripoSR'):
                folders.append(item.name)
        return sorted(folders)
    
    def _discover_samples(self) -> List[Dict]:
        """
        Discover all valid samples.
        
        Returns list of dicts with:
            - sample_id: e.g., '000001_gvxr_processed'
            - energy_folder: e.g., 'TripoSR' or 'TripoSR_0.08'
            - available_angles: List of available angles for this sample
            - ground_truth: Path to ground truth mesh
        """
        samples = []
        
        for energy_folder in self.energy_folders:
            energy_path = self.base_path / energy_folder
            
            if not energy_path.exists():
                continue
            
            # Find all sample folders (XXXXXX_gvxr_processed)
            for sample_dir in sorted(energy_path.iterdir()):
                if not sample_dir.is_dir():
                    continue
                if not sample_dir.name.endswith('_gvxr_processed'):
                    continue
                
                sample_id = sample_dir.name
                
                # Find available angles
                available_angles = self._find_available_angles(sample_dir)
                
                if len(available_angles) < self.n_views:
                    continue
                
                # Find ground truth
                gt_path = self._find_ground_truth(sample_id)
                
                if gt_path is None:
                    warnings.warn(f"No ground truth found for {sample_id}")
                    continue
                
                samples.append({
                    'sample_id': sample_id,
                    'energy_folder': energy_folder,
                    'sample_dir': sample_dir,
                    'available_angles': available_angles,
                    'ground_truth': gt_path
                })
        
        return samples
    
    def _find_available_angles(self, sample_dir: Path) -> List[int]:
        """Find all available angles for a sample."""
        angles = []
        
        for view_dir in sample_dir.iterdir():
            if not view_dir.is_dir():
                continue
            if not view_dir.name.startswith('view_'):
                continue
            
            # Extract angle from folder name (e.g., 'view_-45' -> -45)
            try:
                angle_str = view_dir.name.replace('view_', '')
                angle = int(angle_str)
                
                # Check if mesh exists
                mesh_path = view_dir / '0' / 'mesh.obj'
                if mesh_path.exists():
                    angles.append(angle)
            except ValueError:
                continue
        
        return sorted(angles)
    
    def _find_ground_truth(self, sample_id: str) -> Optional[Path]:
        """Find ground truth mesh for a sample."""
        # Try different possible locations/names
        possible_paths = [
            self.ground_truth_path / sample_id / 'bone.obj',
            self.ground_truth_path / sample_id / 'ground_truth.obj',
            self.ground_truth_path / sample_id / 'mesh.obj',
            self.ground_truth_path / sample_id / 'gt.obj',
            self.ground_truth_path / f'{sample_id}.obj',
            # Extract numeric ID and try
            self.ground_truth_path / sample_id.split('_')[0] / 'bone.obj',
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def _get_mesh_path(self, sample: Dict, angle: int) -> Path:
        """Get path to mesh file for a specific angle."""
        return sample['sample_dir'] / f'view_{angle}' / '0' / 'mesh.obj'
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a sample.
        
        Returns:
            inputs: Tensor of shape (n_views, D, H, W)
            target: Tensor of shape (1, D, H, W)
            metadata: Dict with sample info including selected angles
        """
        sample = self.samples[idx]
        
        # Select angles for this sample
        # Use idx as part of seed for reproducibility but variation across samples
        selection_seed = self.seed + idx
        
        # Filter angle selector to only available angles
        selector = AngleSelector(
            min_separation=self.min_angle_separation,
            available_angles=sample['available_angles']
        )
        
        selected_angles = selector.select_angles(
            self.n_views,
            strategy=self.angle_strategy,
            seed=selection_seed
        )
        
        # Create cache key
        cache_key = f"{sample['energy_folder']}_{sample['sample_id']}_angles{'_'.join(map(str, selected_angles))}"
        
        # Try loading from cache
        if self.cache_voxels:
            cache_input = self.cache_dir / f'{cache_key}_input.npy'
            cache_target = self.cache_dir / f'{cache_key}_target.npy'
            
            if cache_input.exists() and cache_target.exists():
                inputs = np.load(cache_input).astype(np.float32)
                target = np.load(cache_target).astype(np.float32)
                
                if target.ndim == 3:
                    target = target[np.newaxis, ...]
                
                inputs = torch.from_numpy(inputs)
                target = torch.from_numpy(target)
                
                if self.transform:
                    inputs = self.transform(inputs)
                if self.target_transform:
                    target = self.target_transform(target)
                
                metadata = {
                    'sample_id': sample['sample_id'],
                    'energy_folder': sample['energy_folder'],
                    'selected_angles': selected_angles,
                    'idx': idx
                }
                
                return inputs, target, metadata
        
        # Load and voxelize meshes
        if not MESH_UTILS_AVAILABLE:
            raise ImportError("mesh_utils required for mesh loading")
        
        # Get mesh paths for selected angles
        mesh_paths = [self._get_mesh_path(sample, angle) for angle in selected_angles]
        
        # Process input views
        inputs = prepare_multi_view_input(
            mesh_paths,
            resolution=self.resolution,
            align=True,
            fill_interior=self.fill_interior
        )
        
        # Process ground truth
        target_mesh = load_obj_mesh(sample['ground_truth'])
        target = mesh_to_voxels(
            target_mesh,
            resolution=self.resolution,
            fill_interior=self.fill_interior
        )
        
        # Cache if enabled
        if self.cache_voxels:
            np.save(cache_input, inputs)
            np.save(cache_target, target)
        
        # Add channel dimension to target
        target = target[np.newaxis, ...]
        
        inputs = torch.from_numpy(inputs.astype(np.float32))
        target = torch.from_numpy(target.astype(np.float32))
        
        if self.transform:
            inputs = self.transform(inputs)
        if self.target_transform:
            target = self.target_transform(target)
        
        metadata = {
            'sample_id': sample['sample_id'],
            'energy_folder': sample['energy_folder'],
            'selected_angles': selected_angles,
            'idx': idx
        }
        
        return inputs, target, metadata
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata about a sample without loading it."""
        return self.samples[idx]


class TripoSRGVXRDatasetFixedAngles(Dataset):
    """
    Dataset variant where angles are fixed at initialization.
    
    This is more efficient for training as angles don't change between epochs.
    """
    
    def __init__(
        self,
        base_path: Union[str, Path],
        ground_truth_path: Union[str, Path],
        n_views: int = 3,
        resolution: int = 64,
        energy_folders: Optional[List[str]] = None,
        angle_strategy: str = 'random',
        min_angle_separation: int = 45,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_voxels: bool = True,
        fill_interior: bool = True,
        seed: int = 42,
        preselect_angles: bool = True
    ):
        self.base_path = Path(base_path)
        self.ground_truth_path = Path(ground_truth_path)
        self.n_views = n_views
        self.resolution = resolution
        self.angle_strategy = angle_strategy
        self.min_angle_separation = min_angle_separation
        self.transform = transform
        self.target_transform = target_transform
        self.cache_voxels = cache_voxels
        self.fill_interior = fill_interior
        self.seed = seed
        
        # Find energy folders
        if energy_folders is None:
            energy_folders = self._find_energy_folders()
        self.energy_folders = energy_folders
        
        # Create cache directory
        if cache_voxels:
            self.cache_dir = self.base_path / '.cache' / f'res{resolution}_n{n_views}_fixed'
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Discover all samples
        self.samples = self._discover_samples()
        
        # Pre-select angles for each sample
        if preselect_angles:
            self._preselect_all_angles()
        
        print(f"Found {len(self.samples)} samples")
    
    def _find_energy_folders(self) -> List[str]:
        folders = []
        for item in self.base_path.iterdir():
            if item.is_dir() and item.name.startswith('TripoSR'):
                folders.append(item.name)
        return sorted(folders)
    
    def _discover_samples(self) -> List[Dict]:
        samples = []
        
        for energy_folder in self.energy_folders:
            energy_path = self.base_path / energy_folder
            
            if not energy_path.exists():
                continue
            
            for sample_dir in sorted(energy_path.iterdir()):
                if not sample_dir.is_dir():
                    continue
                if not sample_dir.name.endswith('_gvxr_processed'):
                    continue
                
                sample_id = sample_dir.name
                available_angles = self._find_available_angles(sample_dir)
                
                if len(available_angles) < self.n_views:
                    continue
                
                gt_path = self._find_ground_truth(sample_id)
                
                if gt_path is None:
                    continue
                
                samples.append({
                    'sample_id': sample_id,
                    'energy_folder': energy_folder,
                    'sample_dir': sample_dir,
                    'available_angles': available_angles,
                    'ground_truth': gt_path,
                    'selected_angles': None  # Will be filled by _preselect_all_angles
                })
        
        return samples
    
    def _find_available_angles(self, sample_dir: Path) -> List[int]:
        angles = []
        for view_dir in sample_dir.iterdir():
            if not view_dir.is_dir() or not view_dir.name.startswith('view_'):
                continue
            try:
                angle = int(view_dir.name.replace('view_', ''))
                mesh_path = view_dir / '0' / 'mesh.obj'
                if mesh_path.exists():
                    angles.append(angle)
            except ValueError:
                continue
        return sorted(angles)
    
    def _find_ground_truth(self, sample_id: str) -> Optional[Path]:
        possible_paths = [
            self.ground_truth_path / sample_id / 'bone.obj',
            self.ground_truth_path / sample_id / 'ground_truth.obj',
            self.ground_truth_path / sample_id / 'mesh.obj',
            self.ground_truth_path / f'{sample_id}.obj',
        ]
        for path in possible_paths:
            if path.exists():
                return path
        return None
    
    def _preselect_all_angles(self):
        """Pre-select angles for all samples."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        for i, sample in enumerate(self.samples):
            selector = AngleSelector(
                min_separation=self.min_angle_separation,
                available_angles=sample['available_angles']
            )
            
            selected = selector.select_angles(
                self.n_views,
                strategy=self.angle_strategy,
                seed=self.seed + i
            )
            
            sample['selected_angles'] = selected
    
    def _get_mesh_path(self, sample: Dict, angle: int) -> Path:
        return sample['sample_dir'] / f'view_{angle}' / '0' / 'mesh.obj'
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        sample = self.samples[idx]
        selected_angles = sample['selected_angles']
        
        # Cache key
        cache_key = f"{sample['energy_folder']}_{sample['sample_id']}_{'_'.join(map(str, selected_angles))}"
        
        if self.cache_voxels:
            cache_input = self.cache_dir / f'{cache_key}_input.npy'
            cache_target = self.cache_dir / f'{cache_key}_target.npy'
            
            if cache_input.exists() and cache_target.exists():
                inputs = torch.from_numpy(np.load(cache_input).astype(np.float32))
                target = torch.from_numpy(np.load(cache_target).astype(np.float32))
                
                if target.ndim == 3:
                    target = target.unsqueeze(0)
                
                if self.transform:
                    inputs = self.transform(inputs)
                if self.target_transform:
                    target = self.target_transform(target)
                
                return inputs, target, {
                    'sample_id': sample['sample_id'],
                    'energy_folder': sample['energy_folder'],
                    'selected_angles': selected_angles,
                    'idx': idx
                }
        
        if not MESH_UTILS_AVAILABLE:
            raise ImportError("mesh_utils required")
        
        mesh_paths = [self._get_mesh_path(sample, angle) for angle in selected_angles]
        
        inputs = prepare_multi_view_input(
            mesh_paths,
            resolution=self.resolution,
            align=True,
            fill_interior=self.fill_interior
        )
        
        target_mesh = load_obj_mesh(sample['ground_truth'])
        target = mesh_to_voxels(
            target_mesh,
            resolution=self.resolution,
            fill_interior=self.fill_interior
        )
        
        if self.cache_voxels:
            np.save(cache_input, inputs)
            np.save(cache_target, target)
        
        target = target[np.newaxis, ...]
        
        inputs = torch.from_numpy(inputs.astype(np.float32))
        target = torch.from_numpy(target.astype(np.float32))
        
        if self.transform:
            inputs = self.transform(inputs)
        if self.target_transform:
            target = self.target_transform(target)
        
        return inputs, target, {
            'sample_id': sample['sample_id'],
            'energy_folder': sample['energy_folder'],
            'selected_angles': selected_angles,
            'idx': idx
        }


def custom_collate_fn(batch):
    """Custom collate function that handles metadata."""
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    return inputs, targets, metadata


def create_gvxr_data_loaders(
    base_path: Union[str, Path],
    ground_truth_path: Union[str, Path],
    n_views: int = 3,
    batch_size: int = 4,
    resolution: int = 64,
    energy_folders: Optional[List[str]] = None,
    angle_strategy: str = 'random',
    min_angle_separation: int = 45,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders for gvxr dataset.
    """
    # Create full dataset
    full_dataset = TripoSRGVXRDatasetFixedAngles(
        base_path=base_path,
        ground_truth_path=ground_truth_path,
        n_views=n_views,
        resolution=resolution,
        energy_folders=energy_folders,
        angle_strategy=angle_strategy,
        min_angle_separation=min_angle_separation,
        cache_voxels=True,
        seed=seed
    )
    
    # Split dataset
    n_samples = len(full_dataset)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    n_test = n_samples - n_train - n_val
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader, test_loader


# Keep original classes for backward compatibility
class VoxelAugmentation:
    """Data augmentation for 3D voxel data."""
    
    def __init__(
        self,
        rotation: bool = True,
        flip: bool = True,
        noise_std: float = 0.0,
        elastic: bool = False
    ):
        self.rotation = rotation
        self.flip = flip
        self.noise_std = noise_std
        self.elastic = elastic
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.rotation:
            k = np.random.randint(0, 4)
            if k > 0:
                x = torch.rot90(x, k, dims=[-2, -1])
        
        if self.flip:
            if np.random.random() > 0.5:
                x = torch.flip(x, dims=[-1])
            if np.random.random() > 0.5:
                x = torch.flip(x, dims=[-2])
            if np.random.random() > 0.5:
                x = torch.flip(x, dims=[-3])
        
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = torch.clamp(x + noise, 0, 1)
        
        return x


if __name__ == '__main__':
    # Test angle selector
    print("Testing AngleSelector...")
    selector = AngleSelector(min_separation=45)
    
    print(f"Valid angles count: {len(selector.valid_angles)}")
    print(f"Valid angle ranges: {AngleSelector.VALID_RANGES}")
    
    # Test different strategies
    for strategy in ['random', 'uniform', 'fixed']:
        for n_views in [2, 3, 4, 5]:
            angles = selector.select_angles(n_views, strategy=strategy, seed=42)
            print(f"Strategy={strategy}, n_views={n_views}: {angles}")
    
    # Test valid combinations
    valid_combos = selector.get_all_valid_combinations(3, max_combinations=100)
    print(f"\nValid 3-angle combinations: {len(valid_combos)}")
    print(f"First 5: {valid_combos[:5]}")
