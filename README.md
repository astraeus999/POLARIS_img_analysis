# POLARIS_img_analysis

A collection of polarimetric imaging analysis workflows and models aligned with POLARIS benchmark datasets.

## 🚀 Overview

This repository provides a curated suite of image‑analysis approaches designed to work with polarimetric imaging datasets—such as the POLARIS high‑contrast polarimetric dataset for exoplanetary disks. Implemented workflows include:

- **DeepCluster**
- **Diff-SimCLR**
- **DiskDiffusion**
- **SimCLR**
- **VaeImputation**
- **MaskEncoder**

Each offers a self-supervised or generative modeling strategy for representation learning and downstream analysis.

## 🧭 Repository Structure

```
POLARIS_img_analysis/
├── DeepCluster/
├── Diff‑SimCLR/
├── DiskDiffusion/
├── SimCLR/
├── VaeImputation/
├── maskencoder/
├── downstream_tasks_clean/
└── labeled_images_example/
```

## ⚙️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/astraeus999/POLARIS_img_analysis.git
   cd POLARIS_img_analysis
   ```

2. Create a Python environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate
   ```

3. Install dependencies (add a requirements file per module):
   ```bash
   pip install -r requirements.txt
   ```

## 💡 Usage Examples

Each method directory (e.g., `SimCLR/` or `VaeImputation/`) includes:

- Jupyter notebooks or Python scripts demonstrating training and evaluation
- Sample input/output files in `labeled_images_example/`
- Pre‑processing and post‑processing scripts (e.g. for mask encoding in `maskencoder/`)

Example to run SimCLR:

```bash
cd SimCLR
python train_simclr.py   --data_path ../labeled_images_example/   --output_dir ./checkpoints
```

## 🎯 Downstream Tasks

The `downstream_tasks_clean/` module includes evaluation pipelines such as:

- Classification performance (e.g., disk vs. reference star)
- Clustering metrics (e.g., silhouette score, ARI)
- Reconstruction quality (e.g., MSE, SSIM)

## 🧪 Example Dataset (Optional)

`labeled_images_example/` includes subfolders of small sample polarimetric images labeled for testing. These serve as quick validations of each workflow before applying to full-scale benchmark datasets.

## 📚 References

- **POLARIS** benchmark dataset on polarimetric imaging for exoplanet disks and representation learning  
- Community discussions on converting Vectra Polaris TIFF tiles into OME‑TIFF for QuPath workflows

## 🛠️ Roadmap & Contributing

Potential improvements include:

- Integration with Polaris benchmark judging APIs  
- Support for additional self‑supervised encoders (e.g. MoCo, BYOL)  
- Utility scripts for multi-tile OME‑TIFF stitching pipelines  

Contributions welcome via issues or fork‑and‑PR. Please follow existing code style and include corresponding tests where applicable.

## 📄 License

Specify repository license here (e.g. MIT, Apache‑2.0).

## ❓ Contact / Author
@astraeus999
@aistoume
