# BG-LiDAR

Code for PointPillars, BG-PointPillars, Complex YOLO and BG-Complex YOLO on TUMTraf-I and RCooper.

The BG models use a background representation from a fixed roadside LiDAR to help distinguish traffic objects from the static scene. BG-PointPillars takes the current point cloud and a background point cloud as inputs. BG-Complex YOLO uses a background pseudo-image alongside the current-frame pseudo-image.

Both implementations include training, evaluation and visualization scripts. The baseline and BG versions are kept in the same project folder so they can use the corresponding dataset loaders and evaluation tools.

## Downloads

- [TUMTraf-I dataset](https://innovation-mobility.com/en/project-providentia/a9-dataset/) (registration required)
- [TUMTraf development kit](https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit)
- [RCooper dataset and download instructions](https://github.com/AIR-THU/DAIR-RCooper#data-download)

Download the datasets separately. The code uses `A9` for TUMTraf-I and `RC` for RCooper.

Follow the dataset providers' instructions for access and annotation formats. Dataset files and pretrained weights are not included in this code package.

## TUMTraf-I annotations and splits

The repository also includes the converted TUMTraf-I labels and frame indices used in the experiments:

- `ImageSet/new_train.txt` and `ImageSet/new_val.txt`: the modified training and evaluation splits.
- `ImageSet/original_train.txt` and `ImageSet/original_val.txt`: the original split indices.
- `labels/modified/`: converted labels with the three evaluation classes.
- `labels/original/`: converted labels retaining the original classes.

The modified split lists correspond to `new_train_north_south_v2.txt` and `new_val_north_south_v2.txt` in the model dataset metadata. See the [TUMTraf-I extension guide](docs/TUMTraf-I-extension.md) for the existing label-conversion documentation. Download the point clouds and images from the dataset provider.

## Directory structure

```text
BG-PointPillars/
    pcdet/                 Model components, dataset loaders and utilities
    tools/                 Training, prediction and visualization scripts
    tools/cfgs/            Model and dataset configurations
    data/                  Preprocessing scripts
    dataset_support/       Dataset helpers, class names and split lists
    setup.py               OpenPCDet extension setup

BG-Complex-YOLO/
    src/models/            Network definitions
    src/data_process/      Dataset loaders and input preparation
    src/config/            Training settings and model configurations
    src/utils/             Losses, metrics and visualization utilities
    src/                   Training, evaluation and visualization scripts
    dataset/               Preprocessing scripts and dataset metadata
    requirements.txt       Python dependencies
```

## Setup

`BG-PointPillars/` contains the PointPillars implementation and `BG-Complex-YOLO/` contains the Complex YOLO implementation. Both folders include baseline and background-assisted models.

Set up the Python/PyTorch environment and OpenPCDet extensions before training. Additional Complex YOLO dependencies are listed in `BG-Complex-YOLO/requirements.txt`.

Set the dataset and checkpoint locations in the configuration files and the script you plan to run. For PointPillars, check `tools/cfgs/dataset_configs/` and the selected model YAML. For Complex YOLO, check `src/config/train_config.py`, `src/config/train_config_rc.py` and the evaluation script. BG models also require the precomputed background input.

Before starting an experiment:

1. Prepare the point clouds, labels and any calibration files used by the selected loader.
2. Use the provided split lists and check that their frame IDs match the prepared files.
3. For PointPillars, prepare the dataset-info and database files referenced by the dataset YAML.
4. For BG models, prepare the background representation for the same sensor and scene as the current frames.
5. Select the model configuration and set the batch size, learning rate and number of epochs for the experiment.

The split lists and preprocessing helpers are included as supporting files. Downloaded datasets still need to be arranged or converted to the format expected by each implementation.

## Background generation

Generate a separate background for each fixed LiDAR sensor. Use frames from that sensor's training split and keep the background in the coordinate system expected by the dataset loader. The generated background is reused as an additional input for frames from the same sensor.

### Background point clouds for BG-PointPillars

The RCooper generator is `BG-PointPillars/data/RC_dataset/make_RC_background_by_3d_grid.py`. Keep its accompanying `tools.py` in the same directory. The generator reads binary point clouds with four float32 values per point: `x, y, z, intensity`.

The script performs the following steps:

1. Read the training split and shuffle the frame list.
2. Collect a fixed number of frames for each sensor: RC-117, RC-120, RC-136 and RC-139.
3. Apply the coordinate adjustment and region-of-interest filter defined in `tools.py`, then combine the selected point clouds for each sensor.
4. Divide the combined points into a 3D voxel grid.
5. Keep voxels containing at least `background_threshold_cnt` points. Represent each retained voxel by the mean position and intensity of its points.
6. Save one background point cloud per sensor as a binary file.

Set these values before generating backgrounds:

| Setting | Purpose | Current PointPillars script default |
|---|---|---|
| `sample_num` | Number of input frames per sensor | `25` |
| `voxel_size` | Voxel dimensions in metres | `[0.05, 0.05, 0.05]` |
| `background_threshold_cnt` | Minimum accumulated point count per voxel | `15` |
| `boundary` in `tools.py` | Spatial region used for background construction | Set for the target scene |
| `save_path` | Output directory | Set for the prepared dataset |

The threshold counts points accumulated across the selected frames, not the number of distinct frames in which a voxel appears. The current script default uses 25 frames; set `sample_num` to the experiment setting when reproducing results that use 50 frames. Set a random seed before shuffling if the same frame selection is needed across runs.

Run the script with the prepared RCooper dataset root as the working directory. It expects `ImageSets/train.txt` and the point-cloud files under `training/pointcloud/`. Create the output directory specified by `save_path` before running. The script also opens Open3D visualization windows after saving the backgrounds.

Two details in the supplied source need attention before generation:

- The `voxel_grid_139` assignment currently uses `pc_136_overlapped`. Change its input to `pc_139_overlapped` to generate the RC-139 background from the correct sensor.
- Output filenames currently end in `_25_15_roi_100.bin`, while the RCooper loader expects `_25_15.bin`. Align the output names and the names in `get_background()` in `pcdet/datasets/RCooper/RCooper_dataset.py`.

The BG-PointPillars loaders read backgrounds as float32 arrays with four values per point. The RCooper loader selects a background using the sensor ID in the frame name. The TUMTraf-I loader selects separate north/south backgrounds through `get_background()` in `pcdet/datasets/A9/A9_dataset.py`.

For TUMTraf-I, follow the same procedure as RCooper to generate separate backgrounds for the north and south LiDARs. Adjust the frame selection and region of interest for each sensor, and match the output filenames to the A9 loader.

### Background pseudo-images for BG-Complex YOLO

The median-background experiments use a BEV pseudo-image rather than a background point cloud directly. The RCooper script `BG-PointPillars/data/RC_dataset/make_RC_background_by_mean_median.py` builds per-sensor BEV backgrounds and saves mean/median arrays as `.npy` files. Its current `sample_num` is `200`.

Match the generated array filenames and locations to those read by the selected `*_dataset_background_median.py` loader in `BG-Complex-YOLO/src/data_process/`. For the background-point-cloud-derived pseudo-image variant, first prepare the background point cloud, then use the corresponding `*_from_pcd_background.py` pipeline.

## PointPillars and BG-PointPillars

Run commands from `BG-PointPillars/`.

| Task | Script |
|---|---|
| PointPillars training | `tools/prune_train.py` |
| BG-PointPillars training | `tools/prune_train_background_dual_pointnet.py` |
| TUMTraf-I prediction | `tools/test.py` |
| RCooper prediction | `tools/test_RC.py` |
| Visualization | `tools/visualization_for_paper.py` |

Model configurations are under `tools/cfgs/A9_models/` and `tools/cfgs/RC_models/`:

| Dataset / model | Configuration under `tools/cfgs/` |
|---|---|
| TUMTraf-I / PointPillars | `A9_models/pointpillar_new_1_v2_best_2nd.yaml` |
| TUMTraf-I / BG-PointPillars | `A9_models/pointpillar_new_1_v2_dual_pointnet_background_0.005_8_2nd.yaml` |
| RCooper / PointPillars | `RC_models/pointpillar.yaml` |
| RCooper / BG-PointPillars | `RC_models/pointpillar_dual_pointnet_background.yaml` |

Example TUMTraf-I training commands:

```bash
# PointPillars
python tools/prune_train.py --cfg_file tools/cfgs/A9_models/pointpillar_new_1_v2_best_2nd.yaml

# BG-PointPillars
python tools/prune_train_background_dual_pointnet.py --cfg_file tools/cfgs/A9_models/pointpillar_new_1_v2_dual_pointnet_background_0.005_8_2nd.yaml
```

For RCooper, pass the corresponding RC YAML to the baseline or BG training script. Training options such as `--batch_size` and `--epochs` can be supplied on the command line. Check both the script arguments and YAML settings when changing an experiment.

For the foreground-extraction comparison, use `tools/prune_train_foreground_extraction.py` with `tools/cfgs/RC_models/pointpillar_foreground_extraction_octree_65D.yaml`.

### Evaluation

The paper uses the Complex YOLO metric for both detection frameworks. Evaluate PointPillars and BG-PointPillars in two steps:

1. Run `tools/test.py` for TUMTraf-I or `tools/test_RC.py` for RCooper with the matching configuration and checkpoint. Export predictions through `tools/eval_utils/eval_utils.py`, using `--batch_size 1`.
2. From `BG-Complex-YOLO/src/`, run `evaluate_from_pointpillar.py` for TUMTraf-I or `evaluate_from_pointpillar_RC.py` for RCooper.

Set the same prediction directory in the exporter and evaluator. Keep outputs separate for each model and dataset. Both evaluators read `{frame_id}_pointpillar_a9.npy`, including the RCooper evaluator.

For example, export TUMTraf-I baseline predictions from `BG-PointPillars/`:

```bash
python tools/test.py --cfg_file tools/cfgs/A9_models/pointpillar_new_1_v2_best_2nd.yaml --ckpt CHECKPOINT.pth --batch_size 1
```

Replace `CHECKPOINT.pth` with the matching checkpoint. For BG-PointPillars, select the BG YAML and checkpoint. For RCooper, use `tools/test_RC.py` and the RC configuration.

Then run the appropriate metric script from `BG-Complex-YOLO/src/`:

```bash
# TUMTraf-I predictions
python evaluate_from_pointpillar.py

# RCooper predictions
python evaluate_from_pointpillar_RC.py
```

Set the evaluator's dataset, class-name file and prediction directory before running it. The shared metric functions are in `src/utils/evaluation_utils.py`. Use this evaluation flow when comparing PointPillars results with Complex YOLO results from the paper.

## Complex YOLO and BG-Complex YOLO

Run commands from `BG-Complex-YOLO/src/`.

| Dataset / model | Training | Evaluation |
|---|---|---|
| TUMTraf-I / Complex YOLO | `prune_train_a9.py` | `evaluate_with_bev.py` |
| TUMTraf-I / BG-Complex YOLO | `prune_train_a9_background_median.py` | `evaluate_with_bev_median.py` |
| RCooper / Complex YOLO | `prune_train_RC.py` | `evaluate_with_bev_RC.py` |
| RCooper / BG-Complex YOLO | `prune_train_RC_background_median.py` | `evaluate_with_bev_RC_median.py` |

The BG entries in this table use median background pseudo-images. Training settings are defined in `src/config/train_config.py` and `src/config/train_config_rc.py`; network configurations are under `src/config/cfg/`.

Example TUMTraf-I BG training and evaluation:

```bash
python prune_train_a9_background_median.py
# Set the trained checkpoint in the evaluation script, then run:
python evaluate_with_bev_median.py
```

The RCooper baseline script imports `config.train_config`; set its dataset configuration to RCooper before training. The RCooper BG script uses `config.train_config_rc`.

Select the checkpoint in the corresponding evaluation script after training. Keep the baseline and BG checkpoints separate and use the same dataset split and metric settings for comparisons.

Files named `*_from_pcd_background.py` implement the background-point-cloud-derived pseudo-image variant.

## Visualization

For PointPillars and BG-PointPillars, use `tools/visualization_for_paper.py` from `BG-PointPillars/`.

For Complex YOLO and BG-Complex YOLO, run the appropriate script from `BG-Complex-YOLO/src/`:

| Dataset | Baseline | BG model |
|---|---|---|
| TUMTraf-I | `test_a9_baseline.py` | `test_a9_median.py` |
| RCooper | `test_RC_baseline.py` | `test_RC_median.py` |

Set the model configuration, checkpoint, input frames and confidence threshold in the selected script. Use the same frames and display settings when preparing baseline/BG comparisons. Confidence thresholds used for figures can differ from those used to compute AP; keep the evaluation settings fixed when reporting metrics.

## Licenses

The model implementations retain their upstream licenses:

- PointPillars / BG-PointPillars: [Apache License 2.0](BG-PointPillars/LICENSE).
- Complex YOLO / BG-Complex YOLO: [GNU GPL v3](BG-Complex-YOLO/LICENSE).

The existing dataset-extension materials retain the repository's [MIT license](LICENSE). Dataset access and reuse are also subject to the original dataset providers' terms.