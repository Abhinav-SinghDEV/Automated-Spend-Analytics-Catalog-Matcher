# Cross-Enterprise (CPSE) Material Master Deduplication Engine

An AI-powered spend analytics and catalog harmonization pipeline designed to solve data fragmentation and duplicate material entries across public sector enterprises (CPSEs).

## Overview
Industrial procurement suffers from inconsistent naming conventions, typos, and fragmented descriptions across different enterprise resource planning (ERP) systems. This project leverages string similarity metrics and gradient-boosted decision trees to automatically identify and harmonize identical material master records while accounting for real-world data entropy.

## Key Features
* **Advanced Feature Engineering:** Utilizes RapidFuzz similarity metrics (`token_sort_ratio`, `token_set_ratio`, `partial_ratio`) with simulated feature jitter to capture human typing variances.
* **Robust Classification:** Powered by a regularized LightGBM classifier optimized via 5-fold stratified cross-validation.
* **Realistic Data Entropy:** Trained on a 71,400-row pairwise dataset featuring 8% label noise to mirror messy, real-world ERP environments rather than artificially clean benchmarks.
* **Interactive UI:** Includes a Streamlit dashboard (`app.py`) built for live, real-time material matching and batch inference demonstrations.

 ## Architecture:-
<img width="2816" height="1536" alt="System Architecture" src="https://github.com/user-attachments/assets/4dc54a51-99be-4e5e-9261-f47af03a43c0" />

## Model Performance:-
<img width="2816" height="1536" alt="Model Performance " src="https://github.com/user-attachments/assets/badf4dde-38ab-4249-bc9d-12852f23a297" />




## Tech Stack
* **Machine Learning:** LightGBM, Scikit-Learn
* **String Matching:** RapidFuzz
* **Frontend:** Streamlit
* **Data Processing:** Pandas, NumPy

## Project Structure
```text
├── data/
│   └── synthetic/
│       └── synthetic_cpse_catalog.csv  # Master reference catalog (3,200 rows)
├── models/
│   └── lgb_material_harmonizer.pkl       # Saved production model artifact
├── 01_Data_Generation_and_Model_Training.ipynb # Core pipeline notebook
└── app.py                                # Streamlit interactive UI
