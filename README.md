# Hierarchical Object Detection

**Detecting fine-grained classes with a hierarchy-aware DINO**: instead of predicting only leaf classes, the model assigns object queries to a set of **class embeddings for every node in the taxonomy** (leaves *and* parents). At inference time, a query's top-1 node may fall back to a semantically valid ancestor (e.g. *"some Fixed-Wing Fighter"* instead of an uncertain *"F-16"*), and training is supervised by a **Hierarchical Focal Loss (HFL)** that reweights focal-style terms along the semantic path.

Built on top of [MMDetection](https://github.com/open-mmlab/mmdetection).

## Highlights

- **+4.2 hAF1** over a flat DINO baseline on a fine-grained military aircraft dataset (dataset v86, 81 leaf classes in a curated functional taxonomy): 0.786 vs. 0.744, with near-parity on standard flat metrics (hmAP 0.709 vs. 0.748).
- **Much more robust to corruption**: under image corruption stress, hierarchical single-label decoding preserves semantic F1 best (mPC hAF1 **0.601** vs. 0.496 for flat DINO, a relative **+10.5**), with the most stable precision (rPC hAF1 0.770).
- **Zero-shot generalization**: with six held-out leaf classes, hierarchical supervision clearly surpasses the flat baseline in hAF1 and hmAP, producing meaningful ancestor-level fallback predictions for unseen classes.
- **Hierarchy-agnostic**: on a second (geopolitical) hierarchy, the best hierarchical model reaches hAF1 0.742 vs. 0.733 for the flat proxy, confirming the gains come from the supervision mechanism, not a single taxonomy.

## Key results

Main comparison on the held-out test set (soft-binary decoding, decay = 10):

| Model | hmAP | hAR@100 | hAF1 | Node-based hmAP | Node-based hAR@100 | Node-based hAF1 |
|---|---|---|---|---|---|---|
| DINO (flat baseline) | 0.748 | 0.892 | 0.744 | 0.748 | 0.893 | 0.746 |
| DINO-E (multi-label) | 0.728 | 0.892 | 0.723 | 0.729 | 0.894 | 0.723 |
| DINO-E (single-label) | 0.747 | 0.843 | 0.785 | 0.749 | 0.844 | 0.786 |
| DINO-H (multi-label) | 0.639 | 0.862 | 0.650 | 0.656 | 0.863 | 0.660 |
| **DINO-H (single-label)** | **0.709** | 0.780 | **0.786** | **0.720** | 0.781 | **0.791** |

Robustness under image corruption (mean performance under corruption / relative preservation):

| Model | mPC hAF1 | rPC hAF1 |
|---|---|---|
| DINO (flat baseline) | 0.496 | 0.666 |
| DINO-H (multi-label, decay = 10) | 0.332 | 0.590 |
| **DINO-H (single-label, decay = 10)** | **0.601** | 0.767 |
| DINO-H (single-label, decay = 2) | 0.555 | **0.770** |

Incremental ablation (single-label, ∆ relative to the HFL-only DINO-H baseline of 0.709 / 0.780 / 0.786): the **entailment-cone loss** is the strongest individual addition (**+0.025 hmAP, +0.023 hAR@100, +0.007 hAF1**). The full ablation table, the corruption matrices, and the zero-shot setup are in `notebooks/`.

## Repository layout

```
hod/
├── models/
│   ├── dense_heads/        # Embedding-based DINO & RTMDet heads (+ filtered head)
│   ├── layers/             # Embedding classifier over taxonomy nodes
│   ├── losses/             # Hierarchical Focal Loss, entailment-cone, contrastive
│   └── task_modules/       # Hierarchy-aware matching costs
├── datasets/               # Hierarchical COCO API wrappers + builders
├── evaluation/             # HierarchicalCocoMetric (hmAP / hAR@100 / hAF1)
├── training/               # Prototype pre-training
├── hpo/                    # Hyperparameter search helpers
└── utils/                  # HierarchyTree, checkpoint compatibility
configs/hod/                # Experiments: func/area taxonomies, decay sweep, ablations, RTMDet
tools/
├── train.py / test.py      # Standard MMDet entry points (+ prototype pre-training hook)
├── dist_train.sh           # Multi-GPU training
├── dataset_converters/     # Aircraft CSV -> hierarchical COCO JSON
├── analysis_tools/         # Robustness eval, confusion matrix, projection plots
└── misc/                   # Dataset download
docker/                     # PyTorch 2.6 + CUDA 11.8 dev container (mmdet, mmcv, mmengine)
notebooks/                  # Analysis notebooks
```

## Setup

### Prerequisites

- Docker + Docker Compose (with NVIDIA runtime)
- A Kaggle API key for the dataset

### Installation

```bash
git clone https://github.com/Gogonemnem/hierarchical-object-detection.git
cd hierarchical-object-detection

docker compose build
docker compose up -d                    # container name: hod
docker exec -it hod bash
pip install -e .                        # inside the container
```

The container mounts the current directory at `/root/hierarchical-object-detection` and exposes SSH on `${PORT}` (defaults to `2222` in `.env`; set `PORT` in `docker/.env` if 22 conflicts) so you can `ssh`/VSCode straight into the GPU environment.

## Dataset preparation

1. Create a Kaggle API key and export the credentials (inside the container):

   ```bash
   export KAGGLE_USERNAME=<your-username>
   export KAGGLE_KEY=<your-api-key>
   ```

2. Download the military aircraft dataset:

   ```bash
   bash tools/misc/download_aircraft_dataset.sh
   ```

   This fetches the images and `labels_with_split.csv` (with pre-defined `train` / `val` / `test` splits) into `data/aircraft/`.

    > **Dataset version note**: the headline results above (and the decay/ablation sweeps in `configs/`) were all obtained on dataset **v86** (81 leaf classes). The current dataset is **v104** (114 leaf classes), which is what the `aircraft.py` converter now produces — the reported numbers are therefore on the older, smaller set.
    The converter supports two taxonomies:

   - **Functional** – groups aircraft by role (fighters, bombers, UAVs, …).
   - **Area** – groups aircraft by region of origin.

   ```bash
   python tools/dataset_converters/aircraft.py \
       --csv-file data/aircraft/labels_with_split.csv \
       --out-dir data/aircraft
   ```

    Output per split (e.g. `train`): `aircraft_train.json`, `aircraft_hierarchy_function_train.json`, `aircraft_hierarchy_area_train.json`. Each JSON also carries the full taxonomy in its `taxonomy` key. This is the **updated version of the dataset (v104)**, which contains **114 leaf classes** (up from 81 in v86); both taxonomies embed every one of them, so no class is left without hierarchical supervision. The functional taxonomy has **145 categories** (114 leaves + 31 internal nodes, including a `Commercial Aircraft` grouping for the airliners) and the regional (area) taxonomy has **134 categories** (114 leaves + 20 internal nodes). Set `num_classes` in model configs accordingly (114 flat, 145 func, 134 area).

## Training and evaluation

Single GPU:

```bash
python tools/train.py configs/hod/models/hierarchical_dino_func_sweep/hi_dino-r50-hfl10-sep-eucl_improved_8xb2-36e_aircraft-func.py
```

Multi-GPU (the configs above are tuned for `8xb2`, scale batch/epochs via `--cfg-options` on smaller GPUs):

```bash
bash tools/dist_train.sh <config>.py <num-gpus>
```

Evaluation (checkpoints are saved every epoch under `work_dirs/<work-dir>/`):

```bash
python tools/test.py <config>.py work_dirs/<work-dir>/epoch_36.pth
```

`train_all.sh` batch-runs every config in a given directory (set `CONFIG_DIRS`). Additional entry points: `tools/pretrain_prototypes.py` (standalone prototype pre-training) and `tools/tune.py` (MMSearch hyperparameter search).

The full experiment matrices (decay sweeps, ablations, zero-shot, corruption) live in `configs/hod/models/` and `tools/analysis_tools/`.

## License

MIT — see [LICENSE](LICENSE).
