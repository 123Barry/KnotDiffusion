# KnotDiffusion: Fine-tuning SE(3) Diffusion Models for Knotted Protein Backbone Generation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.YOUR_DOI.svg)](https://doi.org/10.5281/zenodo.YOUR_DOI)

<div align="center">
  <img src="./media/Figure_1.png" alt="KnotDiffusion Demo" width="800">
</div>

This repository contains the official implementation of the paper **KnotDiffusion: A Generative Model for De Novo Design of Knotted Proteins**. KnotDiffusion is a specialized diffusion model for generating knotted protein backbones. Built upon the original [SE(3) diffusion model](https://github.com/jasonkyuyim/se3_diffusion), our approach fine-tunes the model on a curated dataset of knotted proteins to enable generation of complex, topologically non-trivial protein structures.

## 🔬 Key Features

- **Specialized Dataset**: Over 200,000 knotted proteins from [KnotProt 2.0](http://knotprot.cent.uw.edu.pl/) and [AlphaKnot 2.0](https://alphaknot.cent.uw.edu.pl/) databases
- **Fine-tuned Model**: Enhanced performance for generating valid and diverse knotted protein backbones

## 📋 Table of Contents

- [🚀 Installation](#-installation)
- [📊 Dataset Setup](#-dataset-setup)
- [🔬 Inference](#-inference)
- [🎯 Fine-tuning](#-fine-tuning)
- [📚 Citation](#-citation)
- [🤝 Acknowledgements](#-acknowledgements)


## 🚀 Installation

We recommend using [Mamba](https://github.com/mamba-org/mamba) for faster dependency resolution.

```bash
# 1. Create environment
mamba env create -f se3.yml

# 2. Activate environment
conda activate se3

# 3. Install KnotDiffusion package
pip install -e .
```

## 📊 Dataset Setup

### Download the Dataset

The dataset is archived on Zenodo with a permanent DOI:

```bash
# Download dataset
wget -O knotted_proteins.tar.gz "https://zenodo.org/uploads/16492608/knotted_proteins.tar.gz"

# Extract dataset
tar -xzvf knotted_proteins.tar.gz

# Preprocess data
python ./data/process_pdb_files.py
```

After preprocessing, your directory structure should look like:

```
.
├── knot_dataset/           # Raw PDB files
│   ├── protein_001.pdb
│   └── ...
├── knot_pkl/              # Processed pickle files
│   ├── protein_001.pkl
│   └── ...
├── knot_metadata.csv      # Dataset metadata
└── ...
```

### Data Sources

Our dataset aggregates high-quality knotted proteins from:
- [KnotProt 2.0](http://knotprot.cent.uw.edu.pl/)
- [AlphaKnot 2.0](https://alphaknot.cent.uw.edu.pl/)

## 🔬 Inference

### Quick Start

Generate knotted protein backbones using our fine-tuned model:

```bash
python experiments/inference_se3_diffusion.py
```

By default, this generates 2,000 protein samples with lengths randomly chosen between 100-500 residues.

### Model Checkpoints

Two model weights are available:

| Model | Path | Description |
|-------|------|-------------|
| **KnotDiffusion** (default) | `weights/finetuned/knot_weights.pth` | Fine-tuned on knotted proteins |
| Original FrameDiff | `weights/original_paper/best_weights.pth` | Original SE(3) diffusion weights |

### Output Structure

Generated samples are saved to `inference_outputs/` with the following structure:

```
inference_outputs/
└── 27D_07M_2025Y_5h_26m_50s/
    ├── inference_conf.yaml         # Configuration used
    ├── length_195_1/               # Sample 1 (195 residues)
    │   ├── bb_traj_1.pdb           # x_{t-1} diffusion trajectory
    │   ├── sample_1.pdb            # Final sample
    │   ├── x0_traj_1.pdb           # Model predictions
    │   └── self_consistency/       # Self consistency results 
    │       ├── esmf/               # ESMFold predictions
    │       ├── seqs/               # ProteinMPNN sequences
    |       ├── parsed_pdbs.jsonl   # Parsed chains for ProteinMPNN
    │       ├── sample_1.pdb        # the copy of final sample
    │       └── sc_results.csv      # Metrics summary
    └── length_356_2/               # Sample 2 (356 residues)
        └── ...
```

### Configuration

Customize inference by editing `config/inference.yaml`:

```yaml
inference:
  weights_path: weights/finetuned/knot_weights.pth  # Model to use
  # ... other parameters
```

## 🎯 Fine-tuning

### Prerequisites

The base model checkpoint (`best_weights.pth`) of the original [Framediff paper](https://arxiv.org/abs/2302.02277) is included in `./weights/original_paper/`.

### Configuration

Ensure correct paths in `config/base.yaml`:

```yaml
data:
  csv_path: ./knot_metadata.csv      # Dataset metadata
  cluster_path: ./train_cluster.txt  # Training clusters

experiment:
  warm_start: ./weights/original_paper/  # Base model directory
```

### Launch Training

```bash
python experiments/train_se3_diffusion.py
```

Training outputs:
- **Checkpoints**: Saved to `ckpt/` directory
- **Evaluations**: Saved to `eval_outputs/` directory

For detailed training information, refer to the original [FrameDiff repository](https://github.com/jasonkyuyim/se3_diffusion).

## 📚 Citation

If you use KnotDiffusion in your research, please cite both our work and the original SE(3) diffusion model:

```bibtex
@unpublished{knotdiffusion_2025,
  title={KnotDiffusion: A Generative Model for De Novo Design of Knotted Proteins},
  author={Wang, Qingquan and Deng, Puqing},
  journal={Manuscript submitted for publication},
  year={2025}
}

@article{yim2023se,
  title={SE(3) diffusion model with application to protein backbone generation},
  author={Yim, Jason and Trippe, Brian L and De Bortoli, Valentin and Mathieu, Emile and Doucet, Arnaud and Barzilay, Regina and Jaakkola, Tommi},
  journal={arXiv preprint arXiv:2302.02277},
  year={2023}
}
```

## 🤝 Acknowledgements

This work builds upon the foundational [SE(3) diffusion model](https://github.com/jasonkyuyim/se3_diffusion) by Yim et al. We thank the authors for their excellent work and open-source implementation. Go give this repos a star if you use this codebase!

We also acknowledge the [KnotProt 2.0](http://knotprot.cent.uw.edu.pl/) and [AlphaKnot 2.0](https://alphaknot.cent.uw.edu.pl/) databases for providing high-quality knotted protein structures.

---