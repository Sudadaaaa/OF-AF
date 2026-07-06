# OF-AF

One-Shot Flow, Any-Time Frame: A Bidirectional Warping Framework for Event-Based Video Frame Interpolation.

## Installation

```bash
conda create -n ofaf python=3.10
conda activate ofaf
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install opencv-contrib-python==4.11.0.86
pip install lpips==0.1.4
pip install tqdm==4.66.4
pip install cupy-cuda12x==13.3.0 -i https://pypi.org/simple/
pip install scikit-image==0.24.0
```

The above is a quick-start script. For the full list of dependencies, see [requirements.txt](requirements.txt).

## Getting Started

### Training

Launch training with the provided script:

```bash
bash train_model.sh
```

Key arguments (edit inside `train_model.sh` or override via command line):

| Argument | Default | Description |
|----------|---------|-------------|
| `--epoch` | 20 | Number of training epochs |
| `--lr` | 2e-4 | Learning rate |
| `--dataset` | gopro | Dataset: gopro or ... |
| `--voxel_bins` | 128 | Event voxel bins |
| `--nb_of_flow` | 16 | Number of flow steps |
| `--batch_size` | 1 | Batch size per GPU |
| `--save_dir` | train | Checkpoint save directory |

Weights are saved to `train/{save_dir}/epoch{N}.pth`.

### Evaluation

A reference evaluation script is provided in `bsergb_eval.py`:

```bash
python bsergb_eval.py --checkpoint train/bsergb.pth
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | train/bsergb.pth | Path to checkpoint |
| `--save_pred` | False | Save predicted frames |
| `--nb_of_flow` | 16 | Number of flow steps |
| `--voxel_bins` | 128 | Event voxel bins |

## Pre-trained Weights

Download the pre-trained checkpoint from [GitHub Releases](https://github.com/Sudadaaaa/OF-AF/releases):

```bash
mkdir -p train
wget https://github.com/Sudadaaaa/OF-AF/releases/download/v1.0/bsergb.pth -O train/bsergb.pth
```

Load the weights:

```python
from models.model import MyModel
import torch, argparse

args = argparse.Namespace(voxel_bins=128, nb_of_flow=16)
model = MyModel(args)
state = torch.load('train/bsergb.pth')['model']
model.load_state_dict(state)
```

## Model

```
models/
├── model.py      # MyModel — top-level architecture
├── encoder.py    # EventsEncoder, ImageEncoder
├── BiFEB.py      # BidirFlow, Query, SepConvGRU
├── BiW.py        # BiW, MaskGuideNet, RefBlock
├── blocks.py     # ResBlock, ResBlockIF, BackwardWarp
└── softsplat.py  # CUDA softmax splatting
```

## Acknowledgements

Based on the paper "One-Shot Flow, Any-Time Frame" (ECCV 2026).
