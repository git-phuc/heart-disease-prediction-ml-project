# Heart Disease Prediction (Dự đoán bệnh tim)

## 1. Giới thiệu đề tài
- **Bài toán:** Phân loại (Classification) – dự đoán nguy cơ mắc bệnh tim trong 10 năm (**TenYearCHD**) dựa trên các đặc trưng lâm sàng/nhân khẩu học.
- **Mục tiêu:**
  - Xây dựng pipeline xử lý dữ liệu và huấn luyện mô hình Machine Learning.
  - Đánh giá mô hình bằng các chỉ số phù hợp cho dữ liệu mất cân bằng (PR-AUC/ROC-AUC, Precision/Recall/F1, Balanced Accuracy, Confusion Matrix).
  - Lưu model tốt và chạy demo inference nhanh.
- **Phạm vi:** Thử nghiệm baseline nhiều mô hình ML, sau đó **tuning** một số mô hình tốt nhất bằng **RandomizedSearchCV**.

---

## 2. Dataset
- **Nguồn dữ liệu:** Kaggle – `D11KS.csv`  
- **Link tải:** https://www.kaggle.com/datasets/phamphucai05/d11ks-csv
- **Note:** Dataset **không đưa trực tiếp lên GitHub**. Xem hướng dẫn tải & đặt file tại: `data/README_DATA.md`.

### 2.1. Bảng mô tả dữ liệu (Data Dictionary)

| STT | Tên cột | Kiểu dữ liệu | Mô tả |
|----:|---------|--------------|------|
| 1 | age | int | Tuổi của bệnh nhân |
| 2 | gender | int (0/1) | Giới tính (0: nữ, 1: nam) |
| 3 | education | int | Trình độ học vấn |
| 4 | currentSmoker | int (0/1) | Có hút thuốc hay không |
| 5 | cigsPerDay | float | Số điếu thuốc hút mỗi ngày |
| 6 | BPMeds | int (0/1) | Có dùng thuốc huyết áp |
| 7 | prevalentStroke | int (0/1) | Tiền sử đột quỵ |
| 8 | prevalentHyp | int (0/1) | Tiền sử tăng huyết áp |
| 9 | diabetes | int (0/1) | Có bệnh tiểu đường |
| 10 | totChol | float | Cholesterol toàn phần |
| 11 | sysBP | float | Huyết áp tâm thu |
| 12 | diaBP | float | Huyết áp tâm trương |
| 13 | BMI | float | Chỉ số khối cơ thể |
| 14 | heartRate | float | Nhịp tim |
| 15 | glucose | float | Glucose trong máu |
| 16 | TenYearCHD | int (0/1) | Nhãn mục tiêu: nguy cơ mắc bệnh tim trong 10 năm |

---

## 3. Pipeline (đúng theo notebook của dự án)
Toàn bộ pipeline được triển khai trong notebook: `demo/CHD_10Y_D11KS.ipynb`.

### 3.1. Chuẩn bị dữ liệu
- Đọc file `D11KS.csv`
- **Target:** `TenYearCHD`
- Làm sạch nhãn:
  - ép kiểu numeric (`to_numeric`)
  - thay `inf/-inf` → `NaN`
  - loại bỏ dòng có target không hợp lệ

### 3.2. Chia train/test
- Chia **80/20** bằng `train_test_split(test_size=0.2, stratify=y, random_state=42)` để giữ tỷ lệ lớp.

### 3.3. Tiền xử lý (Preprocess) bằng `ColumnTransformer`
Dự án dùng pipeline chuẩn của sklearn:
- **Numeric columns** (nếu tồn tại): `age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose`
  - `SimpleImputer(strategy="median")`
  - `StandardScaler()`
- **Categorical columns** (nếu tồn tại): `gender, education, currentSmoker, BPMeds, prevalentStroke, prevalentHyp, diabetes`
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`

### 3.4. Huấn luyện (Train) + xử lý mất cân bằng (Imbalance)
- Mỗi model được gắn vào pipeline theo dạng:
  - `preprocess` → (tuỳ chọn) `RandomOverSampler` → `model`
- Nếu máy có `imbalanced-learn` thì dùng `RandomOverSampler`. Nếu không có thì notebook tự bỏ qua và ưu tiên `class_weight="balanced"` cho các model hỗ trợ.

### 3.5. Đánh giá (Evaluate)
- Đánh giá **OOF (Out-Of-Fold) trên TRAIN** với `StratifiedKFold(n_splits=5)` để giảm bias và tránh leakage.
- Metric dùng trong dự án:
  - **ROC-AUC**, **PR-AUC (Average Precision)**  
  - **Precision, Recall, F1-score**
  - **Balanced Accuracy**
  - **Confusion Matrix**
- Chọn **threshold** dự đoán (không mặc định 0.5):
  - Ở baseline: chọn threshold theo **tối ưu F1** dựa trên Precision–Recall curve.
  - Sau tuning: chọn threshold ưu tiên **Recall** với ràng buộc **Precision tối thiểu** (mặc định 0.15) để tăng khả năng “bắt bệnh”.

### 3.6. Tuning (RandomizedSearchCV)
Sau khi chạy baseline, notebook tuning một số model mạnh:
- **RF (Random Forest)**, **ET (Extra Trees)**, **HGB (HistGradientBoosting)**  
- (Tuỳ chọn) **XGB (XGBoost)** nếu cài `xgboost`
- (Tuỳ chọn) **LGBM (LightGBM)** nếu cài `lightgbm`

### 3.7. Inference
- Notebook lưu model tốt nhất (kèm threshold) bằng `joblib` vào thư mục `saved_models/`.
- Demo inference chạy qua `demo/app.py` (hoặc chạy trực tiếp trong notebook).

---

## 4. Mô hình sử dụng (đúng theo notebook)
### 4.1. Baseline models (OOF TRAIN)
- **LR** – Logistic Regression (baseline mạnh, dễ giải thích, nhanh; có `class_weight="balanced"`)
- **GNB** – Gaussian Naive Bayes (đơn giản, chạy nhanh)
- **KNN** – K-Nearest Neighbors (baseline)
- **SVC (RBF)** – Support Vector Classifier (baseline)
- **DT** – Decision Tree (baseline, dễ diễn giải)
- **RF** – Random Forest (baseline)
- **ET** – Extra Trees (baseline)
- **AdaBoost**, **Gradient Boosting**, **HistGradientBoosting**
- (Tuỳ chọn) **XGBoost**, **LightGBM** nếu có thư viện

### 4.2. Mô hình chọn cuối & lý do
Dự án lưu và so sánh tối thiểu 2 mô hình tuning tốt nhất:
- **RF tuned:** cân bằng tốt giữa Accuracy/Precision/Recall
- **ET tuned:** Recall cao và AUC/PR-AUC rất tốt

> Gợi ý triển khai: nếu ưu tiên **cân bằng và giảm báo động giả**, dùng **RF tuned**.  
> Nếu ưu tiên **bắt bệnh tối đa (Recall cao)**, cân nhắc **ET tuned** với threshold phù hợp.

---

## 5. Kết quả (theo lần chạy notebook)
### 5.1. Kết quả TEST (sau tuning)
**RF (Tuned) – TEST**
- Threshold: **0.2768**
- ROC-AUC: **0.9655**
- PR-AUC: **0.9040**
- Accuracy: **0.9445**
- Precision: **0.8276**
- Recall: **0.8073**
- F1-score: **0.8173**

Confusion Matrix (RF – TEST):

|        | Pred0 | Pred1 |
|--------|------:|------:|
| True0  | 1746  | 55    |
| True1  | 63    | 264   |

**ET (Tuned) – TEST**
- Threshold: **0.1801**
- ROC-AUC: **0.9705**
- PR-AUC: **0.9114**
- Accuracy: **0.9023**
- Precision: **0.6258**
- Recall: **0.9052**
- F1-score: **0.7400**

Confusion Matrix (ET – TEST):

|        | Pred0 | Pred1 |
|--------|------:|------:|
| True0  | 1624  | 177   |
| True1  | 31    | 296   |

### 5.2. Nhận xét
- **ET tuned** có **Recall rất cao** (ít bỏ sót ca bệnh), phù hợp khi ưu tiên sàng lọc.
- **RF tuned** có **Accuracy và Precision cao hơn**, phù hợp khi muốn giảm false positive nhưng vẫn giữ Recall tốt.

---

## 6. Hướng dẫn chạy

### 6.1. Cài môi trường
Python **3.9+**.

```bash
pip install -r requirements.txt
```

### 6.2. Chuẩn bị dữ liệu
1) Tải `D11KS.csv` từ Kaggle  
2) Đặt file vào:
```
data/D11KS.csv
```

### 6.3. Chạy train + evaluate + tuning (Notebook)
Mở notebook và Run All:
```bash
jupyter notebook demo/CHD_10Y_D11KS.ipynb
```

Sau khi chạy xong, notebook sẽ:
- in bảng kết quả baseline + tuning
- đánh giá cuối trên TEST
- lưu model vào `saved_models/` (ví dụ: `CHD_RF_tuned.joblib`, `CHD_ET_tuned.joblib`) và in ra threshold tương ứng

### 6.4. Chạy demo / inference nhanh
Sau khi đã có model trong `saved_models/`:
```bash
python demo/app.py
```

Nếu `demo/app.py` yêu cầu đường dẫn model, hãy chỉnh biến `MODEL_PATH` trong file để trỏ tới:
- `saved_models/CHD_RF_tuned.joblib` hoặc
- `saved_models/CHD_ET_tuned.joblib`

---

## 7. Cấu trúc thư mục dự án
```text
app/        -> (khuyến nghị) tách code từ notebook thành các script train/predict/preprocess/utils
demo/       -> notebook + script demo chạy nhanh
data/       -> chỉ chứa data mẫu nhỏ hoặc file hướng dẫn tải data
reports/    -> báo cáo (PDF/Docx)
slides/     -> slide thuyết trình (PPTX/PDF)
saved_models/ -> (tự sinh sau khi chạy notebook) model .joblib + threshold
requirements.txt
README.md
.gitignore
```

---

## 8. Tác giả
- **Họ tên:** Phạm Đình Phúc
- **Mã SV:** 12423063
- **Lớp:** 124231
- **GitHub:** https://github.com/git-phuc
