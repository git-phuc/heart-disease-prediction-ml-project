# app/ (Modular Notebooks)

Thư mục `app/` chứa các notebook đã **chia nhỏ** từ notebook gốc để dễ bảo trì:

- `0_setup.ipynb`: imports, seed, optional libs
- `1_load_clean_split.ipynb`: load data + clean + train/test split
- `2_eda.ipynb`: EDA (tuỳ chọn)
- `3_preprocess_helpers.ipynb`: ColumnTransformer + helpers
- `4_baseline_oof.ipynb`: baseline OOF cho nhiều model
- `5_tuning.ipynb`: RandomizedSearchCV tuning
- `0_test_and_save.ipynb`: test evaluation + save `.joblib`
- `7_inference.ipynb`: demo inference (tuỳ chọn)

Chạy nhanh: mở `demo/00_RUN_ALL.ipynb` và **Run All**.
