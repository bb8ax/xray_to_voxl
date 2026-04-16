"""
Mesh Processing Utilities for TripoSR Fusion

This module handles:
1. Loading mesh files (OBJ, STL, PLY, etc.) from TripoSR and ground truth
2. Converting meshes to voxel grids with consistent normalization
3. Normalizing and aligning meshes to a common coordinate system
4. Preparing data for the 3D U-Net
5. Converting neural network output back to mesh with proper scale

IMPORTANT: All meshes are normalized to a canonical space [-0.5, 0.5]^3
before voxelization to ensure consistent comparison between predictions
and ground truth.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict
import warnings

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    warnings.warn("trimesh not installed. Install with: pip install trimesh")

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not installed. Install with: pip install scipy")


# Supported mesh file formats
SUPPORTED_FORMATS = ['.obj', '.stl', '.ply', '.off', '.glb', '.gltf']


def load_mesh(filepath: Union[str, Path]) -> 'trimesh.Trimesh':
    """
    Load a mesh file (supports OBJ, STL, PLY, OFF, GLB, GLTF).
    
    Args:
        filepath: Path to mesh file
        
    Returns:
        trimesh.Trimesh object
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required for mesh loading")
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Mesh file not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        warnings.warn(f"File format {suffix} may not be fully supported. "
                      f"Supported formats: {SUPPORTED_FORMATS}")
    
    # Load mesh - force='mesh' ensures we get a single Trimesh object
    # even if the file contains a scene with multiple objects
    mesh = trimesh.load(str(filepath), force='mesh')
    
    # Handle case where mesh might be empty or invalid
    if mesh.vertices.shape[0] == 0:
        raise ValueError(f"Loaded mesh has no vertices: {filepath}")
    
    return mesh


def load_obj_mesh(filepath: Union[str, Path]) -> 'trimesh.Trimesh':
    """
    Load an OBJ mesh file (typically from TripoSR output).
    Alias for load_mesh() for backward compatibility.
    
    Args:
        filepath: Path to OBJ file
        
    Returns:
        trimesh.Trimesh object
    """
    return load_mesh(filepath)


def load_stl_mesh(filepath: Union[str, Path]) -> 'trimesh.Trimesh':
    """
    Load an STL mesh file (common format for ground truth).
    Alias for load_mesh() for clarity.
    
    Args:
        filepath: Path to STL file
        
    Returns:
        trimesh.Trimesh object
    """
    return load_mesh(filepath)


class NormalizationParams:
    """
    Stores normalization parameters to allow inverse transformation.
    
    This is crucial for converting neural network output back to
    the original coordinate system of the ground truth mesh.
    """
    def __init__(self, center: np.ndarray, scale: float):
        self.center = center  # Original centroid
        self.scale = scale    # Scale factor applied
    
    def to_dict(self) -> Dict:
        return {
            'center': self.center.tolist(),
            'scale': float(self.scale)
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'NormalizationParams':
        return cls(
            center=np.array(d['center']),
            scale=d['scale']
        )


def normalize_mesh(
    mesh: 'trimesh.Trimesh',
    target_center: Optional[np.ndarray] = None,
    target_scale: float = 1.0,
    return_params: bool = False
) -> Union['trimesh.Trimesh', Tuple['trimesh.Trimesh', NormalizationParams]]:
    """
    Normalize mesh to fit within a unit cube centered at origin.
    
    The normalization process:
    1. Compute the centroid of the mesh
    2. Center the mesh at origin by subtracting centroid
    3. Scale to fit within [-0.5, 0.5]^3 (or target_scale)
    
    Args:
        mesh: Input trimesh object
        target_center: Target center point (default: origin)
        target_scale: Target scale factor (default: 1.0 means [-0.5, 0.5]^3)
        return_params: If True, also return normalization parameters
        
    Returns:
        Normalized mesh copy, and optionally NormalizationParams
    """
    mesh = mesh.copy()
    
    # Store original parameters for inverse transform
    original_centroid = mesh.centroid.copy()
    
    # Center the mesh at origin
    mesh.vertices -= original_centroid
    
    # Compute scale to fit in unit cube
    bounds = mesh.bounds
    max_extent = np.max(bounds[1] - bounds[0])
    
    if max_extent > 0:
        # Scale to fit in [-0.5, 0.5] range, then apply target_scale
        scale_factor = target_scale / max_extent
        mesh.vertices *= scale_factor
    else:
        scale_factor = 1.0
    
    # Move to target center if specified
    if target_center is not None:
        mesh.vertices += target_center
    
    if return_params:
        params = NormalizationParams(
            center=original_centroid,
            scale=scale_factor
        )
        return mesh, params
    
    return mesh


def denormalize_mesh(
    mesh: 'trimesh.Trimesh',
    params: NormalizationParams
) -> 'trimesh.Trimesh':
    """
    Reverse the normalization to restore original scale and position.
    
    Args:
        mesh: Normalized mesh
        params: NormalizationParams from normalize_mesh()
        
    Returns:
        Mesh in original coordinate system
    """
    mesh = mesh.copy()
    
    # Reverse scale
    if params.scale != 0:
        mesh.vertices /= params.scale
    
    # Reverse center offset
    mesh.vertices += params.center
    
    return mesh


def mesh_to_voxels(
    mesh: 'trimesh.Trimesh',
    resolution: int = 64,
    padding: float = 0.1,
    fill_interior: bool = True
) -> np.ndarray:
    """
    Convert a mesh to a 3D voxel grid.
    
    Args:
        mesh: Input trimesh object
        resolution: Voxel grid resolution (resolution x resolution x resolution)
        padding: Padding around the mesh (as fraction of grid size)
        fill_interior: If True, fill the interior of the mesh
        
    Returns:
        3D numpy array of shape (resolution, resolution, resolution)
        with values in [0, 1] representing occupancy
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required for voxelization")
    
    # Normalize mesh to fit in unit cube
    mesh = normalize_mesh(mesh)
    
    # Create voxel grid
    # Expand bounds slightly for padding
    voxel_size = (1.0 + 2 * padding) / resolution
    
    # Use trimesh's voxelization
    try:
        voxels = mesh.voxelized(pitch=voxel_size)
        voxel_grid = voxels.matrix.astype(np.float32)
        
        # Resize to exact resolution if needed
        if voxel_grid.shape != (resolution, resolution, resolution):
            voxel_grid = resize_voxel_grid(voxel_grid, (resolution, resolution, resolution))
        
        # Fill interior if requested
        if fill_interior:
            voxel_grid = fill_voxel_interior(voxel_grid)
            
    except Exception as e:
        warnings.warn(f"Voxelization failed: {e}. Using point sampling fallback.")
        voxel_grid = mesh_to_voxels_sampling(mesh, resolution)
    
    return voxel_grid


def mesh_to_voxels_sampling(
    mesh: 'trimesh.Trimesh',
    resolution: int = 64,
    num_samples: int = 100000
) -> np.ndarray:
    """
    Alternative voxelization using point sampling.
    More robust for complex or non-watertight meshes.
    
    Args:
        mesh: Input mesh
        resolution: Output resolution
        num_samples: Number of points to sample on mesh surface
        
    Returns:
        Voxel grid as numpy array
    """
    mesh = normalize_mesh(mesh)
    
    # Sample points on mesh surface
    try:
        points, _ = trimesh.sample.sample_surface(mesh, num_samples)
    except:
        # Fallback to using vertices directly
        points = mesh.vertices
    
    # Create empty voxel grid
    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    
    # Convert points to voxel indices
    # Points are in [-0.5, 0.5], map to [0, resolution-1]
    indices = ((points + 0.5) * resolution).astype(int)
    indices = np.clip(indices, 0, resolution - 1)
    
    # Fill voxels
    for idx in indices:
        voxel_grid[idx[0], idx[1], idx[2]] = 1.0
    
    # Dilate slightly to fill gaps
    if SCIPY_AVAILABLE:
        voxel_grid = ndimage.binary_dilation(voxel_grid, iterations=1).astype(np.float32)
    
    return voxel_grid


def resize_voxel_grid(
    voxels: np.ndarray,
    target_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Resize a voxel grid to target shape using trilinear interpolation.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for voxel resizing")
    
    from scipy.ndimage import zoom
    
    factors = [t / s for t, s in zip(target_shape, voxels.shape)]
    resized = zoom(voxels, factors, order=1)  # order=1 for bilinear
    
    # Threshold back to binary
    resized = (resized > 0.5).astype(np.float32)
    
    return resized


def fill_voxel_interior(voxels: np.ndarray) -> np.ndarray:
    """
    Fill the interior of a voxelized surface.
    """
    if not SCIPY_AVAILABLE:
        return voxels
    
    # Use binary fill holes
    filled = ndimage.binary_fill_holes(voxels).astype(np.float32)
    return filled


def align_meshes(
    meshes: List['trimesh.Trimesh'],
    method: str = 'centroid'
) -> List['trimesh.Trimesh']:
    """
    Align multiple meshes to a common reference frame.
    
    Args:
        meshes: List of trimesh objects
        method: Alignment method ('centroid', 'bounds', or 'icp')
        
    Returns:
        List of aligned mesh copies
    """
    if len(meshes) == 0:
        return []
    
    aligned = []
    
    if method == 'centroid':
        # Align by centering all meshes
        for mesh in meshes:
            m = mesh.copy()
            m.vertices -= m.centroid
            aligned.append(m)
            
    elif method == 'bounds':
        # Align by bounding box centers
        for mesh in meshes:
            m = mesh.copy()
            center = (m.bounds[0] + m.bounds[1]) / 2
            m.vertices -= center
            aligned.append(m)
            
    elif method == 'icp':
        # Iterative Closest Point alignment
        # Use first mesh as reference
        reference = meshes[0].copy()
        reference.vertices -= reference.centroid
        aligned.append(reference)
        
        for mesh in meshes[1:]:
            m = mesh.copy()
            m.vertices -= m.centroid
            
            # ICP alignment (if available)
            try:
                matrix, transformed, cost = trimesh.registration.icp(
                    m.vertices, reference.vertices, max_iterations=100
                )
                m.vertices = transformed
            except:
                warnings.warn("ICP alignment failed, using centroid alignment")
            
            aligned.append(m)
    else:
        raise ValueError(f"Unknown alignment method: {method}")
    
    return aligned


def prepare_multi_view_input(
    mesh_paths: List[Union[str, Path]],
    resolution: int = 64,
    align: bool = True,
    fill_interior: bool = True
) -> np.ndarray:
    """
    Prepare multiple TripoSR meshes as a stacked voxel input.
    
    Args:
        mesh_paths: List of paths to OBJ files
        resolution: Voxel grid resolution
        align: Whether to align meshes before voxelization
        fill_interior: Whether to fill interior of voxels
        
    Returns:
        Numpy array of shape (N, resolution, resolution, resolution)
        where N is the number of meshes
    """
    # Load meshes
    meshes = [load_obj_mesh(p) for p in mesh_paths]
    
    # Align if requested
    if align and len(meshes) > 1:
        meshes = align_meshes(meshes, method='centroid')
    
    # Voxelize each mesh
    voxels = []
    for mesh in meshes:
        voxel_grid = mesh_to_voxels(mesh, resolution=resolution, fill_interior=fill_interior)
        voxels.append(voxel_grid)
    
    # Stack along first dimension
    return np.stack(voxels, axis=0)


def voxels_to_mesh(
    voxels: np.ndarray,
    threshold: float = 0.5,
    smoothing_iterations: int = 2,
    norm_params: Optional[NormalizationParams] = None
) -> 'trimesh.Trimesh':
    """
    Convert a voxel grid back to a mesh using marching cubes.
    
    The output mesh is in normalized space [-0.5, 0.5]^3 unless
    norm_params is provided to transform back to original space.
    
    Args:
        voxels: 3D numpy array of occupancy values
        threshold: Isosurface threshold
        smoothing_iterations: Number of Laplacian smoothing iterations
        norm_params: If provided, denormalize to original coordinate system
        
    Returns:
        trimesh.Trimesh object
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required for mesh extraction")
    
    try:
        from skimage import measure
        
        # Apply marching cubes
        verts, faces, normals, _ = measure.marching_cubes(
            voxels, level=threshold, spacing=(1.0, 1.0, 1.0)
        )
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        
        # Normalize vertices to [-0.5, 0.5]^3 (same as input normalization)
        # Marching cubes outputs in [0, resolution], we need [-0.5, 0.5]
        resolution = voxels.shape[0]
        mesh.vertices = mesh.vertices / resolution - 0.5
        
        # Smooth if requested
        if smoothing_iterations > 0:
            trimesh.smoothing.filter_laplacian(mesh, iterations=smoothing_iterations)
        
        # Denormalize to original space if params provided
        if norm_params is not None:
            mesh = denormalize_mesh(mesh, norm_params)
        
        return mesh
        
    except ImportError:
        warnings.warn("scikit-image not installed. Using trimesh voxel export.")
        
        # Use trimesh's built-in functionality
        voxel_grid = trimesh.voxel.VoxelGrid(voxels > threshold)
        return voxel_grid.marching_cubes


def save_mesh(mesh: 'trimesh.Trimesh', filepath: Union[str, Path], file_format: str = 'obj'):
    """
    Save a mesh to file.
    
    Args:
        mesh: trimesh object to save
        filepath: Output path
        file_format: Output format ('obj', 'stl', 'ply', etc.)
    """
    mesh.export(str(filepath), file_type=file_format)


class MeshProcessor:
    """
    Convenience class for processing TripoSR meshes and ground truth.
    
    Handles the complete pipeline:
    1. Load mesh (OBJ, STL, PLY, etc.)
    2. Normalize to canonical space
    3. Voxelize
    4. Convert predictions back to mesh
    5. Optionally restore original scale
    
    Example:
        processor = MeshProcessor(resolution=64)
        
        # Process ground truth (STL) and store normalization params
        gt_voxels, gt_params = processor.process_with_params("bone.stl")
        
        # Process TripoSR outputs
        input_voxels = processor.process_multi_view(["view_0.obj", "view_45.obj", "view_90.obj"])
        
        # After neural network prediction...
        # pred_voxels = model(input_voxels)
        
        # Convert prediction back to mesh in original coordinate system
        pred_mesh = processor.voxels_to_mesh(pred_voxels, norm_params=gt_params)
        pred_mesh.export("prediction.stl")
    """
    
    def __init__(
        self,
        resolution: int = 64,
        fill_interior: bool = True,
        align_method: str = 'centroid'
    ):
        self.resolution = resolution
        self.fill_interior = fill_interior
        self.align_method = align_method
        self._last_norm_params = None
    
    def process_single(
        self, 
        mesh_path: Union[str, Path],
        return_params: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, NormalizationParams]]:
        """
        Process a single mesh to voxels.
        
        Args:
            mesh_path: Path to mesh file (OBJ, STL, PLY, etc.)
            return_params: If True, also return normalization parameters
            
        Returns:
            Voxel grid, and optionally NormalizationParams
        """
        mesh = load_mesh(mesh_path)
        mesh_norm, params = normalize_mesh(mesh, return_params=True)
        
        voxels = mesh_to_voxels(
            mesh_norm, 
            resolution=self.resolution,
            fill_interior=self.fill_interior
        )
        
        self._last_norm_params = params
        
        if return_params:
            return voxels, params
        return voxels
    
    def process_with_params(
        self, 
        mesh_path: Union[str, Path]
    ) -> Tuple[np.ndarray, NormalizationParams]:
        """
        Process mesh and return normalization parameters.
        Shortcut for process_single(..., return_params=True).
        """
        return self.process_single(mesh_path, return_params=True)
    
    def process_multi_view(self, mesh_paths: List[Union[str, Path]]) -> np.ndarray:
        """Process multiple meshes and stack them."""
        return prepare_multi_view_input(
            mesh_paths,
            resolution=self.resolution,
            align=True,
            fill_interior=self.fill_interior
        )
    
    def voxels_to_mesh(
        self, 
        voxels: np.ndarray, 
        threshold: float = 0.5,
        norm_params: Optional[NormalizationParams] = None
    ) -> 'trimesh.Trimesh':
        """
        Convert voxels back to mesh.
        
        Args:
            voxels: 3D voxel grid from neural network output
            threshold: Isosurface threshold (0.5 for sigmoid output)
            norm_params: If provided, transform mesh to original coordinate system
            
        Returns:
            trimesh.Trimesh object
        """
        # Use stored params if none provided
        if norm_params is None:
            norm_params = self._last_norm_params
        
        return voxels_to_mesh(voxels, threshold=threshold, norm_params=norm_params)
    
    @property
    def last_norm_params(self) -> Optional[NormalizationParams]:
        """Get the normalization params from the last processed mesh."""
        return self._last_norm_params


def find_ground_truth_file(
    directory: Union[str, Path],
    sample_id: str,
    preferred_names: List[str] = None
) -> Optional[Path]:
    """
    Find ground truth mesh file in a directory.
    Supports multiple formats (OBJ, STL, PLY).
    
    Args:
        directory: Directory to search
        sample_id: Sample identifier (e.g., "000001_gvxr_processed")
        preferred_names: List of preferred filenames to search for
        
    Returns:
        Path to ground truth file, or None if not found
    """
    directory = Path(directory)
    
    if preferred_names is None:
        preferred_names = ['bone', 'ground_truth', 'gt', 'mesh', 'model']
    
    # Search in sample subdirectory
    sample_dir = directory / sample_id
    
    search_dirs = [sample_dir, directory]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        # Try each preferred name with each supported format
        for name in preferred_names:
            for ext in SUPPORTED_FORMATS:
                path = search_dir / f"{name}{ext}"
                if path.exists():
                    return path
        
        # Also try sample_id as filename
        for ext in SUPPORTED_FORMATS:
            path = directory / f"{sample_id}{ext}"
            if path.exists():
                return path
    
    return None


if __name__ == '__main__':
    # Test with a simple mesh
    print("Testing mesh processing utilities...")
    print(f"Supported formats: {SUPPORTED_FORMATS}")
    
    if TRIMESH_AVAILABLE:
        # Create a test mesh (sphere)
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=50.0)  # 50mm radius sphere
        mesh.vertices += np.array([100, 200, 300])  # Offset from origin
        
        print(f"\nOriginal mesh:")
        print(f"  Centroid: {mesh.centroid}")
        print(f"  Bounds: {mesh.bounds}")
        print(f"  Extent: {mesh.bounds[1] - mesh.bounds[0]}")
        
        # Normalize and get parameters
        normalized, params = normalize_mesh(mesh, return_params=True)
        print(f"\nNormalized mesh:")
        print(f"  Centroid: {normalized.centroid}")
        print(f"  Bounds: {normalized.bounds}")
        print(f"\nNormalization params:")
        print(f"  Original center: {params.center}")
        print(f"  Scale factor: {params.scale}")
        
        # Voxelize
        voxels = mesh_to_voxels(normalized, resolution=64)
        print(f"\nVoxel grid:")
        print(f"  Shape: {voxels.shape}")
        print(f"  Occupied voxels: {np.sum(voxels > 0)}")
        
        # Convert back to mesh (in normalized space)
        reconstructed_norm = voxels_to_mesh(voxels)
        print(f"\nReconstructed (normalized):")
        print(f"  Vertices: {len(reconstructed_norm.vertices)}")
        print(f"  Bounds: {reconstructed_norm.bounds}")
        
        # Convert back to original space
        reconstructed_orig = voxels_to_mesh(voxels, norm_params=params)
        print(f"\nReconstructed (original space):")
        print(f"  Centroid: {reconstructed_orig.centroid}")
        print(f"  Bounds: {reconstructed_orig.bounds}")
        
        # Test MeshProcessor
        print("\n--- Testing MeshProcessor ---")
        processor = MeshProcessor(resolution=64)
        
        # Save test mesh as STL
        test_stl_path = Path("/tmp/test_mesh.stl")
        mesh.export(str(test_stl_path))
        print(f"Saved test mesh to: {test_stl_path}")
        
        # Process STL file
        voxels2, params2 = processor.process_with_params(test_stl_path)
        print(f"Processed STL: voxels shape={voxels2.shape}, occupied={np.sum(voxels2 > 0)}")
        
        # Simulate neural network output and convert back
        pred_mesh = processor.voxels_to_mesh(voxels2, norm_params=params2)
        print(f"Prediction mesh centroid: {pred_mesh.centroid}")
        
        # Clean up
        test_stl_path.unlink()
        
    else:
        print("trimesh not available for testing")
