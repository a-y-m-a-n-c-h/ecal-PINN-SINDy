# ECAL Calibration with Discrete-Time SINDy

This repository contains a standalone Python script that uses **Sparse Identification of Non-linear Dynamics (SINDy)** to model and forecast ECAL crystal calibrations based on laser monitoring data. It implements a **discrete-time** one-step map:

> cₖ₊₁ = f(cₖ, ℓₖ)

where:
- **c** is the (scaled) calibration ratio  
- **ℓ** is the (scaled) delivered integrated luminosity  

The script trains a polynomial model with STLSQ regularization and compares **two** forecast modes:  
1. **Teacher-forcing** (Case 1): uses ground-truth cₖ at each step  
2. **Free-roll** (Case 2): feeds model predictions back into subsequent steps  

---

## 📦 Requirements

- Python 3.8 or later  
- Install dependencies:
  ```bash
  pip install pysindy numpy pandas scikit-learn matplotlib

## 🚀 Running the Script

To train and forecast using the SINDy model, run the following command from the project root:

```bash
python sindy_ecal.py \
  --data_root "./data" \
  --crystal_id 54000 \
  --train_year 2016 \
  --test_years 2017 2018 \
  --poly_deg 2 \
  --threshold 0.05
