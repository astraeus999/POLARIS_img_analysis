# POLARIS_img_analysis

A collection of polarimetric imaging analysis workflows and models aligned with POLARIS benchmark datasets.
Find the paper at: https://arxiv.org/abs/2506.03511
Find dataset at: https://zenodo.org/records/15493277

## 🚀 Overview

This repository provides a curated suite of image‑analysis approaches designed to work with polarimetric imaging datasets—such as the POLARIS high‑contrast polarimetric dataset for exoplanetary disks. Implemented unsupervised models workflows include:


- **Diff-SimCLR** - our proposed latent-enhanced contrasive model for disk representation learning
- **DiskDiffusion** - latent features learning and embedding process
- **DeepCluster** - Upgraded to Python 3.11 From: [Paper]https://arxiv.org/abs/1807.05520
- **SimCLR** From: [1] Supervised Contrastive Learning. [Paper](https://arxiv.org/abs/2004.11362)  [2] A Simple Framework for Contrastive Learning of Visual Representations. [Paper](https://arxiv.org/abs/2002.05709)  
- **MaskEncoder** From: [Paper]https://arxiv.org/abs/2111.06377

Together, with VAE for center circustellar disk reconstruction:
- **VaeImputation** - our VAE for validation of RDI quality obtained from Diff-SimCLR+downstream tasks

Each offers a self-supervised or generative modeling strategy for representation learning.

## 🧭 Repository Structure

```
POLARIS_img_analysis/
├── Diff‑SimCLR/
│   ├── main_simclr.py
│   ├── command_diffsimclr.sh
│   └── ...
├── DeepCluster/
│   ├── clustering.py
│   ├── command.sh
│   └── ...
├── DiskDiffusion/
│   ├── diskdiffusion.py
│   ├── models/
│   └── requirements.txt
├── SimCLR/
│   └── ...
├── VaeImputation/
│   ├── readme.md
│   ├── I_tot_data_example/
│   ├── assets/
│   ├── checkpoints/
│   ├── images/
│   └── requirements.txt
├── maskencoder/
│   ├── train.py
│   ├── command_masencoder.sh
│   └── ...
├── downstream_tasks_clean/
│   ├── GMM.py
│   ├── KNN.py
│   ├── MLP.py
│   ├── spectral.py
│   ├── svc_lin.py
│   └── random_forest.py
├── labeled_images_example/
│   ├── features/ - Latent features extracted from DiskDiffusion
│   └── Sample training images
├── README.md
└── environment.yaml

## ⚙️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/astraeus999/POLARIS_img_analysis.git
   cd POLARIS_img_analysis
   ```

2. Create a Python environment from environment.yaml:
   ```bash
   conda env create -f environment.yaml
   conda activate polaris_img_analysis
   ```

## 💡 Usage Examples

Example to run Diff-simCLR:

A. Train conditional DDPM:

```bash
cd DiskDiffusion
python train.py --epoch 200 --train_data_path ./labeled_images_example --test_data_path ./labeled_images_example
```

B. Sampling and get latent features:

```bash
cd DiskDiffusion
python Sampling.py 
```

This should generate a folder `./labeled_images_example/features/` with latent features extracted from DiskDiffusion; and a .csv file for features mapping.

C. train Diff-SimCLR model:

```bash
cd Diff-SimCLR
python train.py --epoch 200 --train_data_path ./data_folder --test_data_path ./data_folder --resume
```

D: Validate and get representation learning results:

```bash
cd Diff-SimCLR
python test_new.py
```

A 32D representation will be gernerated for each image and saved in a .npy file.

## 💡 Other Generative Models: SimCLR, maskencoder, DeepCluster

Following the `command.sh` file in each subfolder for training and testing.

## 🎯 Downstream Tasks

The `downstream_tasks_clean/` module downstream ML evaluation pipelines

- Both supervised and unsupervised ML methods


## 🧪 Example Dataset (Optional)

`labeled_images_example/` includes subfolders of small sample polarimetric images labeled. The corresponding label is shown on their file name. These serve as quick validations of each workflow before applying to full-scale benchmark datasets.
`features/` contains latent features extracted from DiskDiffusion for downstream tasks.

## 📚 References

- **POLARIS** benchmark dataset on polarimetric imaging for exoplanet disks and representation learning  
If you find this repository useful, please cite our paper:
```
@misc{cao2025polarishighcontrastpolarimetricimaging,
      title={POLARIS: A High-contrast Polarimetric Imaging Benchmark Dataset for Exoplanetary Disk Representation Learning}, 
      author={Fangyi Cao and Bin Ren and Zihao Wang and Shiwei Fu and Youbin Mo and Xiaoyang Liu and Yuzhou Chen and Weixin Yao},
      year={2025},
      eprint={2506.03511},
      archivePrefix={arXiv},
      primaryClass={astro-ph.EP},
      url={https://arxiv.org/abs/2506.03511}, 
}
```

## 📄 License

C.C BY-NC-SA 4.0 License. 

## ❓ Contact / Author
@astraeus999
@aistoume
