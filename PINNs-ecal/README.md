# Physics-Informed Seq2Seq for ECAL Calibration Forecasting

This project implements a physics-informed Seq2Seq model for forecasting ECAL crystal calibration under radiation damage, using a combination of data-driven learning and physical priors.

## 🚀 Setup Instructions

1. **Install Jupyter Notebook**
   ```bash
   sudo apt update
   sudo apt install jupyter-notebook
   ```

2. **Create and activate the environment (GPU version)**
   ```bash
   conda env create -f fair_gpu.yml
   conda activate fair_gpu
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter-notebook
   ```

4. **Navigate to and run** `main.ipynb`.

## 🛠️ Key Modifications

### 1. `seq2seq_train.py`
- Introduced **physics-informed loss** \(\mathcal{L}_{\text{physics}}\) using:
  $$
  \mathcal{L}_{\text{physics}} = \frac{1}{T} \sum_{t=1}^{T} \left( \frac{d\hat{C}}{dt} + k_{\text{dmg}} L(t)\hat{C}(t) - k_{\text{rec}}(C_{\infty} - \hat{C}(t)) \right)^2
  $$
- Added **learnable physical parameters**: `k_dmg`, `k_rec`, and `C_inf`.
- Applied physics loss only after unnormalizing the predicted calibration values.

### 2. `ecal_dataset_prep.py`
- Augmented input features to include:
  - `Δt` (delta time between measurements)
  - `ΔL(t)` (delta luminosity)
  - `ΔC(t)` (delta calibration)

## ⚠️ Normalization Pitfall & Solution

A key issue encountered was that:
- The model was trained on normalized features using `StandardScaler`.
- However, the physics loss depends on **real-world (unscaled)** values for physical consistency.

To resolve this:
- We **unnormalized** the predicted calibration values using the saved `mean` and `std` from the `StandardScaler` before computing the physics loss:
  ```python
  C_pred_unscaled = C_pred_scaled * sigma + mean
  ```

This ensured that the gradient from the physics loss propagated meaningfully during training.
