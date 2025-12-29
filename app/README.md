# app/ (Modular Notebooks)

Thư mục `app/` chứa các notebook đã **chia nhỏ** từ notebook gốc để dễ bảo trì:

- `00_setup.ipynb`: imports, seed, optional libs
- `01_load_clean_split.ipynb`: load data + clean + train/test split
- `02_eda.ipynb`: EDA (tuỳ chọn)
- `03_preprocess_helpers.ipynb`: ColumnTransformer + helpers
- `04_baseline_oof.ipynb`: baseline OOF cho nhiều model
- `05_tuning.ipynb`: RandomizedSearchCV tuning
- `06_test_and_save.ipynb`: test evaluation + save `.joblib`
- `07_inference.ipynb`: demo inference (tuỳ chọn)

Chạy nhanh: mở `demo/00_RUN_ALL.ipynb` và **Run All**.
