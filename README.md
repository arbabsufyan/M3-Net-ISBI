
# M3-Net: Multi-Scale Nuclei Segmentation in Breast Cancer Histopathology

This repository contains the official implementation of **M3-Net**, a multi-scale nuclei segmentation model that uses contextual patches and attention mechanisms for breast cancer histopathology images.

The code corresponds to the method described in:

> **M3-Net: A Multi-Scale Nuclei Segmentation Model for Breast Cancer Histopathology Using Contextual Patches and Attention Mechanism**
> Presented at **IEEE ISBI 2025**.

---

## 🔍 Overview

M3-Net is designed for robust nuclei segmentation by:

* Extracting **multi-scale contextual patches** (large, medium, small) from each image.
* Using a shared **VGG16-based encoder** for all scales.
* Fusing features across scales with **channel attention**.
* Decoding features with a **U-Net–style decoder** to produce segmentation masks.
* Optionally applying **watershed-based instance separation** on the predicted masks.

The provided implementation is in a single Jupyter notebook and is configured to run easily on **Kaggle** (multi-GPU via `tf.distribute.MirroredStrategy`) or any local machine with GPU support.

---

## 📁 Repository Contents

Typical structure:

```text
├── m3-net-isbi2025.ipynb   # Main notebook: data loading, model, training, evaluation
└── README.md               # This file
```

All helper functions (patch extraction, resizing, model definition, post-processing, etc.) are defined inside the notebook.

---

## 🧪 Dataset

By default, the notebook is written to work with the **MoNuSAC** nuclei segmentation dataset on Kaggle.

Expected structure:

```text
MoNuSac/
├── images/         # H&E image tiles
└── binary_masks/   # Corresponding binary nuclei masks (same names as images)
```

In the notebook you will see:

```python
# Load and preprocess training data
train_folder = "/kaggle/input/monusac-public/MoNuSac"
train_images, train_masks = load_and_preprocess_data(
    train_folder,
    "images",
    "binary_masks"
)
```

To use **your own dataset**:

1. Organize images and masks into two folders (e.g., `images`, `masks`).
2. Make sure image–mask filenames correspond.
3. Change `train_folder`, and the two subfolder names (`"images"`, `"binary_masks"`) to match your structure.
4. Ensure masks are binary (nuclei vs background) or adapt the preprocessing for multiclass masks.

---

## ⚙️ Requirements

Tested with:

* Python 3.8+
* TensorFlow 2.x (with Keras)
* CUDA-enabled GPU (optional but recommended)
* Libraries imported in the notebook:

  * `numpy`
  * `opencv-python` (`cv2`)
  * `matplotlib`
  * `scikit-learn`
  * `scikit-image`
  * `seaborn`

Example installation (conda):

```bash
conda create -n m3net python=3.9
conda activate m3net

pip install tensorflow-gpu
pip install numpy opencv-python matplotlib scikit-learn scikit-image seaborn
```

> On Kaggle, most of these packages are already available.

---

## 🚀 How to Run

### Option 1: Kaggle (recommended)

1. Create a new **Kaggle Notebook**.

2. Upload `m3-net-isbi2025.ipynb` or copy its contents.

3. Add the **MoNuSAC** dataset (or your own) as a dataset input.

4. Make sure the dataset path in the cell:

   ```python
   train_folder = "/kaggle/input/monusac-public/MoNuSac"
   ```

   matches the mounted dataset path.

5. Enable **GPU** (or multi-GPU if available).

6. Run all cells in order:

   * Data loading and patch generation
   * Multi-scale patch extraction
   * Model definition (VGG16 backbone + attention + decoder)
   * Training (`model.fit(...)`)
   * Evaluation, qualitative results, and instance post-processing

During training:

* The best model is saved using `ModelCheckpoint` to:

  ```python
  filepath = "/kaggle/working/Binary_model_ER_IHC.keras"
  ```

  You can change this path if needed.

### Option 2: Local Jupyter / Colab

1. Install dependencies and clone/download this repository.

2. Place your dataset in a local folder and update `train_folder` accordingly.

3. Start Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open `m3-net-isbi2025.ipynb` and run the cells sequentially.

If you do not have multiple GPUs, `MirroredStrategy` will still work with a single GPU or fall back to CPU.

---

## 📊 Outputs

The notebook will:

* Train M3-Net and report training/validation:

  * Loss
  * Dice / IoU or related segmentation metrics
* Save the best model as `.keras`.
* Plot example:

  * Input histopathology patches (multi-scale)
  * Ground-truth masks
  * Predicted masks
* Optionally perform **watershed-based instance segmentation** to separate touching nuclei and visualize contours.

---

## 📚 Citation

If you use this code, model, or any part of the workflow in your research, please cite:

```bibtex
@inproceedings{sufyan2025m3,
  title={M3-Net: A Multi-Scale Nuclei Segmentation Model for Breast Cancer Histopathology Using Contextual Patches and Attention Mechanism},
  author={Sufyan, Arbab and Fauzi, Mohammad Faizal Ahmad and Kuan, Wong Lai},
  booktitle={2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI)},
  pages={1--4},
  year={2025},
  organization={IEEE}
}
```
