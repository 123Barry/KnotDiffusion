# KnotDiffusion: Fine-tuning SE(3) Diffusion Models for Knotted Protein Backbone Generation

This repository contains the official implementation for **KnotDiffusion**, a project focused on generating knotted protein backbones. This work is an extension and application of the original **[SE(3) diffusion model](https://github.com/jasonkyuyim/se3_diffusion)**, first presented in the paper "[SE(3) diffusion model with application to protein backbone generation](https://arxiv.org/abs/2302.02277)".

Our core contribution is the fine-tuning of the original model on a specialized, high-quality dataset of knotted proteins, enabling the generation of complex and topologically non-trivial protein structures.

## Citation

If you use KnotDiffusion, our fine-tuned models, or the associated dataset in your research, we kindly ask that you cite **both** our future work and the original SE(3) diffusion model paper.

```bibtex
@article{yim2023se,
  title={SE(3) diffusion model with application to protein backbone generation},
  author={Yim, Jason and Trippe, Brian L and De Bortoli, Valentin and Mathieu, Emile and Doucet, Arnaud and Barzilay, Regina and Jaakkola, Tommi},
  journal={arXiv preprint arXiv:2302.02277},
  year={2023}
}

@artile{knotdiffusion_2025,
  title={KnotDiffusion: A Generative Model for De Novo Design of Knotted Proteins},
  author={Wang, Qingquan and Deng, Puqing},
  journal={Biomacromolecules},
  year={2025}
}

## Core Differences from the Original Work

This project extends the original SE(3) diffusion model with the following key contributions:

* **Specialized Dataset**: We introduce and utilize a high-quality dataset of over 200,000 knotted proteins, aggregated from the [KnotProt 2.0](http://knotprot.cent.uw.edu.pl/) and [AlphaKnot 2.0](https://alphaknot.cent.uw.edu.pl/) databases. This provides a focused training corpus for proteins with complex topologies.

* **Fine-tuned Model**: We provide a model checkpoint, fine-tuned on our specialized dataset, which demonstrates improved performance and stability in generating valid and diverse knotted protein backbones.

## Table of Contents

- [Installation](#installation)
  - [Third party source code](#third-party-source-code)
- [Dataset Setup](#dataset-setup)
  - [Download the Dataset](#download-the-dataset)
  - [Data Sources and Citation](#data-sources-and-citation)
- [Inference](#inference)
  - [Generating Knotted Proteins](#generating-knotted-proteins)
- [Fine-tuning](#fine-tuning)
  - [Download the Base Model](#download-the-base-model)
  - [Launching Fine-tuning](#launching-fine-tuning)
  - [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)


## Installation

We highly recommend using [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) or installing Mamba into your existing Conda environment for significantly faster dependency resolution.

1.  **Create the environment with Mamba.** The following command uses `mamba` to create a new environment named `se3` and install all the necessary dependencies from the provided `se3.yml` file.

    ```bash
    mamba env create -f se3.yml
    ```

2.  **Activate the environment.** Once the installation is complete, activate the newly created environment. Note: environment activation still uses the `conda` command.

    ```bash
    conda activate se3
    ```

3.  **Install the KnotDiffusion package.** Finally, install the KnotDiffusion codebase as an editable package. This allows you to make changes to the source code that will be immediately effective.

    ```bash
    pip install -e .
    ```

After these steps, your environment will be fully set up to run the experiments.


## Dataset Setup

This project requires a specialized dataset of knotted proteins, which is not included directly in this repository due to its size. The dataset must be downloaded and set up manually by following the steps below.

### Download the Dataset

The dataset is permanently archived on **Zenodo** to ensure long-term availability and is assigned a Digital Object Identifier (DOI). We strongly recommend using the DOI to access the dataset's record page for the most stable download link.

**DOI:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.YOUR_DOI.svg)](https://doi.org/10.5281/zenodo.YOUR_DOI)
*(Note: Please replace `YOUR_DOI` with the actual DOI number you receive after publishing your dataset on Zenodo.)*

You can download and extract the dataset using the following commands in your terminal:

```bash
# Step 1: Download the dataset archive from the Zenodo link.
wget -O knotted_proteins.tar.gz "[https://zenodo.org/uploads/16492608/knotted_proteins.tar.gz](https://zenodo.org/uploads/16492608/knotted_proteins.tar.gz)"

# Step 2: Uncompress the archive.
# This command will create the ./knot_dataset/ directory in the project root.
tar -xzvf knotted_proteins.tar.gz

# Step3: Preprocess the Data
# Run the provided script to process the raw .pdb files into a more efficient .pkl format for faster loading during training. This script will create a ./knot_pkl/ directory and a ./knot_metadata.csv file that tracks the processed samples.
python ./data/process_pdb_files.py

# Final Directory Structure
After completing all the steps above, your project directory should look like this:
.
├── knot_dataset/
│   ├── protein_001.pdb
│   ├── protein_002.pdb
│   └── ... (many .pdb files)
│
├── knot_pkl/
│   ├── protein_001.pkl
│   ├── protein_002.pkl
│   └── ... (many .pkl files)
│
├── data/
│   └── process_pdb_files.py
│
├── ...


## Inference

This section describes how to generate *de novo* knotted protein backbones using the provided fine-tuned KnotDiffusion model.

### Model Checkpoints

We provide two sets of model weights within this repository. The fine-tuned KnotDiffusion model is used by default for inference.

The directory structure for the weights is as follows:
```bash
weights/
├── finetuned/
│   └── knot_weights.pth      # (Default) Fine-tuned for knotted proteins
└── original_paper/
    └── best_weights.pth      # Original weights from the FrameDiff paper

KnotDiffusion Weights (finetuned/knot_weights.pth): This is the default checkpoint, fine-tuned on our specialized knotted proteins dataset. It is recommended for generating novel knotted structures.

Original FrameDiff Weights (original_paper/best_weights.pth): These are the original pre-trained weights from the "SE(3) diffusion model" paper, included here for reproducibility and comparison.

Generating Knotted Proteins
The main inference script is experiments/inference_se3_diffusion.py, which utilizes Hydra for configuration.

To run inference, execute the following command:
python experiments/inference_se3_diffusion.py

By default, this will use the fine-tuned knot_weights.pth and generate 2,000 protein backbone samples. In our modified sampling scheme, each sample's length is randomly chosen from a uniform distribution between 100 and 500 residues.

Output Structure
Generated samples and their analysis will be saved to the inference_outputs/ directory by default.

Configuration
You can customize the inference process by editing config/inference.yaml. To use a different set of model weights (for example, to use the original FrameDiff paper's weights), simply modify the weights_path field:

inference:
    weights_path: <path>

inference_outputs/
└── 27D_07M_2025Y_11h_57m_50s          # Date and time of the inference run.
    ├── inference_conf.yaml            # Config used during inference.
    ├── length_195_1/                  # Sample 1, with a length of 195
    │   ├── sample_1.pdb               # Final generated backbone sample
    │   ├── self_consistency/          # Self-consistency analysis results
    │   │   ├── esmf/                  # ESMFold predictions
    │   │   │   ├── sample_0.pdb
    │   │   │   └── ...
    │   │   ├── parsed_pdbs.jsonl      # Parsed chains for ProteinMPNN
    │   │   ├── sc_results.csv         # Summary of self-consistency metrics
    │   │   └── seqs/
    │   │       └── sample_1.fa        # ProteinMPNN designed sequences
    │   ├── bb_traj_1.pdb              # Diffusion trajectory of the backbone
    │   └── x0_traj_1.pdb              # Trajectory of the model's x_0 prediction
    └── length_356_2/                  # Sample 2, with a length of 356
        └── ...                        # (and so on for all 2,000 samples)


## Fine-tuning

Instead of training from scratch, this project focuses on **fine-tuning** the pre-trained SE(3) diffusion model on our specialized knotted proteins dataset. This allows for more efficient training while leveraging the powerful, generalized features learned by the original model.

---

### Base Model for Fine-tuning

Fine-tuning starts from the pre-trained weights of the original **[SE(3) diffusion model](https://github.com/jasonkyuyim/se3_diffusion)**. For your convenience, these weights are already included in this repository.

The base model checkpoint, `best_weights.pth`, is located in the `./weights/original_paper/` directory, and the fine-tuning configuration is pre-set to use this checkpoint as a warm start. No download is required.

---

### Launching the Fine-tuning Run

The configuration for fine-tuning is primarily controlled by `config/base.yaml`.

#### Configuration

Ensure the following paths in `config/base.yaml` are correctly set up. The `warm_start` option points to the directory containing the pre-trained model checkpoint, which will be loaded at the beginning of the training run.

```yaml
# In config/base.yaml
data:
  # Path to the metadata CSV generated during data preprocessing.
  csv_path: ./knot_metadata.csv

  # Path to the training cluster file.
  cluster_path: ./train_cluster.txt

experiment:
  # Directory to warm-start from. This loads the original FrameDiff weights.
  warm_start: ./weights/original_paper/


Launching the Run
Once the configuration is set, start the fine-tuning process by running the main training script:
python experiments/train_se3_diffusion.py

Outputs and Evaluation
The training script will periodically save model checkpoints to the ckpt/ directory and run intermittent evaluations, saving generated samples to eval_outputs/.

The output structure and detailed evaluation metrics are consistent with the original project. For a comprehensive overview of the training outputs, please refer to the Training section of the original **[FrameDiff repository](https://github.com/jasonkyuyim/se3_diffusion)**.