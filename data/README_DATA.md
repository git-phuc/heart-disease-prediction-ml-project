# README_DATA

## 1. Giới thiệu Dataset

Dataset được sử dụng trong dự án Machine Learning này nhằm phục vụ cho quá trình
phân tích dữ liệu, tiền xử lý, huấn luyện và đánh giá mô hình học máy.

Dữ liệu được cung cấp công khai trên nền tảng Kaggle, đảm bảo tính minh bạch
và khả năng tái sử dụng cho mục đích học tập và nghiên cứu.

- Nguồn dữ liệu: Kaggle  
- Định dạng: CSV  
- Mục đích: Huấn luyện và đánh giá mô hình Machine Learning  

Link dataset:  
https://www.kaggle.com/datasets/phamphucai05/d11ks-csv

---

## 2. Cách tải Dataset (Cách 1: Tải thủ công)

Thực hiện theo các bước sau:

1. Truy cập link dataset ở trên.
2. Đăng nhập tài khoản Kaggle.
3. Nhấn nút Download để tải dữ liệu.
4. Giải nén file .zip vừa tải về.
5. Sao chép file .csv vào thư mục data/ của project.

Cấu trúc thư mục sau khi tải dữ liệu:

```text
project_root/
│
├── data/
│   └── d11ks.csv
├── src/
├── notebooks/
└── README_DATA.md


---

## 3. Cách tải Dataset bằng Kaggle API (Khuyến nghị)

### Bước 1: Cài đặt Kaggle API

pip install kaggle

### Bước 2: Tạo Kaggle API Token

1. Truy cập https://www.kaggle.com/account
2. Chọn Create New API Token
3. File kaggle.json sẽ được tải về máy

### Bước 3: Cấu hình Kaggle API

Windows:
C:\Users\<username>\.kaggle\kaggle.json

Linux / macOS:
~/.kaggle/kaggle.json

### Bước 4: Tải dataset

kaggle datasets download -d phamphucai05/d11ks-csv

Giải nén dữ liệu:

unzip d11ks-csv.zip -d data/

---

## 4. Lưu ý khi sử dụng Dataset

- Không chỉnh sửa dữ liệu gốc trước khi sao lưu.
- Cần thực hiện tiền xử lý dữ liệu trước khi huấn luyện mô hình.
- Dataset chỉ sử dụng cho mục đích học tập và nghiên cứu.

---

## 5. Ghi chú cho báo cáo

Ảnh minh họa cần bổ sung trong báo cáo:
- Ảnh trang Kaggle của dataset
- Ảnh cấu trúc thư mục project sau khi tải dữ liệu
