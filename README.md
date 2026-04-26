# X-ray to Voxel: 3D Bone Reconstruction from Multi-View X-ray Images

3D volumetric bone reconstruction from synthetic multi-view X-ray images. The full pipeline is:

1. **X-ray Simulation** (gvxr) - Generate synthetic X-ray projections from bone + flesh STL meshes at multiple gantry angles
2. **Keypoint Detection** (MMPose) - Detect hand bone keypoints in each X-ray image
3. **Segmentation** (SAM) - Use keypoint prompts to segment bone regions from the X-ray images
4. **Per-View 3D Mesh** (TripoSR) - Reconstruct a 3D mesh from each segmented single-view X-ray
5. **Multi-View Fusion** (3D U-Net) - Fuse multiple per-view voxelized meshes into a clean volumetric output

## Project Structure

```
xray_to_voxl/
|-- data/
|   |-- dataset_v2.py          # Dataset with angle combination sampling
|   \-- mesh_utils.py          # Mesh loading, voxelization, alignment utilities
|-- dataset/
|   \-- Hand_Imaging_to_SAM_Dataset_generation.ipynb
|       # End-to-end dataset generation: gvxr -> MMPose -> SAM -> TripoSR
|-- models/
|   \-- unet3d.py              # 3D U-Net and multi-scale variant
|-- tsr/                       # TripoSR inference modules (from stabilityai/TripoSR)
|   |-- system.py
|   |-- utils.py
|   |-- bake_texture.py
|   \-- models/                # Transformer, tokenizers, renderer, isosurface
|-- mmpose.tar.gz              # MMPose source (extract before use, see below)
|-- dataset_gvxr.py            # GVXR dataset loader with angle selection
|-- train_gvxr.py              # Training script (CLI)
|-- train_notebook.ipynb       # Training notebook (interactive, recommended)
|-- test_inference_metric.ipynb # Inference and metric evaluation notebook
|-- prepare_gvxr_data.py       # Convert gvxr views to SAX-NeRF pickle format
|-- losses.py                  # Loss functions (Dice, Focal, Tversky, IoU, etc.)
|-- metrics.py                 # Evaluation metrics (Chamfer, PSNR, SSIM, Dice, IoU)
|-- examples.py                # Dataset structure documentation and examples
|-- run.py                     # CLI for TripoSR single-image inference
\-- requirements.txt
```

## Installation

### Prerequisites

- Python 3.9 (tested with 3.9 + CUDA 11.8)
- CUDA-capable GPU (12 GB+ VRAM recommended)
- The locally installed CUDA major version must match the PyTorch CUDA version

### Steps

Look up the `Hand_Imaging_to_SAM_Dataset_generation.ipynb` for command lines.

1. Clone the repository:
   ```bash
   git clone https://github.com/bb8ax/xray_to_voxl.git
   cd xray_to_voxl
   ```

2. Extract mmpose:
   ```bash
   tar xzf mmpose.tar.gz
   ```

3. Install PyTorch with CUDA 11.8:
   ```bash
   pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
   ```

4. Install MMPose and dependencies:
   ```bash
   pip install -r ./mmpose/requirements.txt
   pip install -U openmim
   pip install numpy==1.26.4
   pip install mmengine==0.10.3
   mim install "mmcv>=2.1.0"
   mim install "mmdet>=3.1.0"
   mim install mmpose
   ```

5. Install SAM:
   ```bash
   pip install segment_anything
   ```

6. Install gvxr (for X-ray simulation):
   ```bash
   pip install gvxr
   ```

7. Install TripoSR dependencies:
   ```bash
   pip install --upgrade setuptools
   pip install pybind11 scikit_build_core "cmake>=3.15"
   pip install -r requirements.txt
   pip install torchmcubes --no-build-isolation
   ```

   If `torchmcubes` fails to compile with CUDA:
   ```bash
   pip uninstall torchmcubes
   pip install git+https://github.com/tatsy/torchmcubes.git
   ```

8. Pin numpy version (required for compatibility):
   ```bash
   pip install numpy==1.26.4
   ```

## Files Not Included

The following files are required but not included in this repository.

### Bone and Flesh STL Models

- **Bone meshes**: `Bone_V1.stl` through `Bone_V5.stl` (5 hand bone models)
- **Flesh meshes**: `Flesh_V4_bool.stl`, `Flesh_V5_bool.stl`, etc.
- These files are used in `Hand_Imaging_to_SAM_Dataset_generation.ipynb` (for X-ray simulation), `train_notebook.ipynb` and `train_gvxr.py` 
-  Place in the project root directory. These are your own 3D hand models in STL format, after receiving the mesh files from the authors.

### SAM Checkpoint

- **File**: `sam_vit_h.pth` (~2.5 GB)
- For SAM neural network in: `Hand_Imaging_to_SAM_Dataset_generation.ipynb`
- Download from [SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints). Update `CHECKPOINT_PATH` in the notebook.

### MMPose Hand Keypoint Checkpoint

- **File**: `rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth`
- For mmpose network in: `Hand_Imaging_to_SAM_Dataset_generation.ipynb`
- Download from [MMPose model zoo](https://mmpose.readthedocs.io/en/latest/model_zoo/hand_2d_keypoint.html). Place in the project root.

### Fusion network checkpoint

- **File**: '0044_v3_r192_E0.06\best.pt'
- For volumetric fusion network in: `test_inference_metric.ipynb`
- Provided upon request.

### TripoSR Pretrained Weights

- **Files**: `config.yaml`, `model.ckpt` (~2.5 GB)
- For wide use in this tool, find HuggingFace `stabilityai/TripoSR`
- Or download it automatically on first run via `TSR.from_pretrained()`. Cached in `~/.cache/huggingface/`.


### Generated Outputs (Upon request > 2Gb)

These are produced by the pipeline and not checked in:

- `./New_Result/` - Masks, segmented images, TripoSR meshes from the dataset generation step
- `./output_gvxr_*/` - Raw gvxr simulation outputs (TIFF/PNG X-ray images)
- `./outputs/*/best.pt` - Trained 3D U-Net checkpoints (~90 MB each)
- `.cache/` - Cached voxelized meshes (auto-generated on first dataset load)
- `./test_results/` - CSV metric results from inference


## Usage

### Step 1: Generate Dataset

Open `dataset/Hand_Imaging_to_SAM_Dataset_generation.ipynb` and configure:

- `DATASET_DIR` - path to input X-ray images (or use gvxr to generate synthetic ones)
- `CHECKPOINT_PATH` - path to `sam_vit_h.pth`
- `checkpoint_file` - path to the MMPose hand keypoint model
- Bone and flesh STL paths in the `bone` and `flesh` lists

The notebook runs the full pipeline: gvxr simulation at multiple angles, MMPose keypoint detection, SAM segmentation, and TripoSR mesh reconstruction. Output meshes are saved to `./New_Result/TripoSR_{energy}/{sample}/view_{angle}/0/mesh.obj`.

Key parameters to configure:
- `source_position`, `detector_position` - X-ray source/detector distance (cm)
- `ray_energy` - X-ray energy in MeV (e.g., 0.06, 0.08)
- `viewing_angles` - list of gantry angles to simulate
- `noise` - noise severity for realistic degradation (0 = clean)

### Step 2: Train the 3D U-Net

1. **Using the notebook** (recommended):

Open `train_notebook.ipynb` and configure:

```python
BASE_PATH = "./New_Result"       # Contains TripoSR_{energy} folders with OBJ meshes
GT_PATH = "."                    # Contains Bone_V{1-5}.stl ground truth files
ENERGY_LEVEL = "0.06"           # Energy subfolder to use
N_VIEWS = 3                      # Number of input views (each needs a separate model)
N_COMBINATIONS = 100             # Angle combinations per bone model
MIN_ANGLE_SEPARATION = 45        # Minimum degrees between selected views
RESOLUTION = 64                  # Voxel grid resolution (64, 128, or 192)
USE_LIGHT_MODEL = True           # True for GPUs with <= 12 GB VRAM
```

2. **Using the CLI**:

```bash
python train.py \
    --base_path ./New_Result \
    --ground_truth_path . \
    --n_views 3 \
    --resolution 192 \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-3
```

Each different `N_VIEWS` value requires a separate trained model. Checkpoints are saved to `./outputs/{timestamp}/best.pt`.

### Step 3: Inference and Evaluation

Open `test_inference_metric.ipynb` and configure:

```python
CHECKPOINT_PATH = "./outputs/.../best.pt"  # Your trained model
TEST_MODE = 3            # 1=individual, 2=multiple samples, 3=random angle combos
TEST_OBJ_DIR = "./New_Result/TripoSR_figure_0.06_.../000001_gvxr_processed"
GROUND_TRUTH_PATH = "./Bone_V5.stl"
N_VIEWS = 3              # Must match the trained model
RESOLUTION = 192         # Must match the trained model
```

Metrics computed: Dice, IoU, PSNR, SSIM, Chamfer Distance, RMSE. Results are saved as CSV to `./test_results/`.

## Changing the Dataset

To use different bone/anatomy models:

1. Replace the STL files (`Bone_V*.stl`, `Flesh_V*_bool.stl`) with your own meshes.
2. In `Hand_Imaging_to_SAM_Dataset_generation.ipynb`, update the `bone` and `flesh` lists and gvxr material parameters (`setMixture`, `setDensity`, `setCompound`).
3. If not using hand bones, you may need a different keypoint model and prompt strategy in place of MMPose + hand_prompts.
4. In `data/dataset.py`, update `VALID_ANGLES` if your views use different angles.


## Examples
1. Download the dataset from 10.5281/zenodo.19798835, and unzip it to source folder.
2. Move the test_inference_metric_example.ipynb into source folder.
3. Run and change the settings inside for different outputs.

## Data availablilty
The synthetic X-ray and mesh dataset generated in this study has been deposited in the Zenodo repository under DOI 10.5281/zenodo.19798835. The dataset includes one hand bone 3D mesh model (Bone_V1, Flesh_V1) which is publicly available. The remaining four hand bone mesh models (Bone_V2, Flesh_V2 through V5) are not publicly available as they are derived from commercially available samples that require consent for distribution. Access to these additional meshes can be obtained by contacting the corresponding author; they will be distributed under restricted access after completion of a data use agreement stipulating that the data will not be redistributed and will be used solely for reproducing the results of this study. Access to these additional meshes can be obtained by contacting the corresponding author (bb8ax@virginia.edu). Requests will be responded to within 14 business days. Upon approval, a data use agreement must be completed stipulating that the data will not be redistributed and will be used solely for reproducing and extending the results of this study. Once access is granted, the data will remain available for the duration of the research project. These restrictions apply to all requestors equally regardless of institutional affiliation. 

## Citation

```BibTeX
@article{TripoSR2024,
  title={TripoSR: Fast 3D Object Reconstruction from a Single Image},
  author={Tochilkin, Dmitry and Pankratz, David and Liu, Zexiang and Huang, Zixuan and Letts, Adam and Li, Yangguang and Liang, Ding and Laforte, Christian and Jampani, Varun and Cao, Yan-Pei},
  journal={arXiv preprint arXiv:2403.02151},
  year={2024}
}

@misc{mmpose2020,
  title={OpenMMLab Pose Estimation Toolbox and Benchmark},
  author={MMPose Contributors},
  howpublished={\url{https://github.com/open-mmlab/mmpose}},
  year={2020}
}

@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Dollar, Piotr and Girshick, Ross},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}

@article{vidal2021gvirtualxray,
  title={gVirtualXRay: Virtual X-Ray Imaging Library on GPU},
  author={Vidal, Franck P. and Villard, Pierre-Fr{\'e}d{\'e}ric},
  journal={SoftwareX},
  volume={16},
  pages={100834},
  year={2021},
  publisher={Elsevier}
}

```
