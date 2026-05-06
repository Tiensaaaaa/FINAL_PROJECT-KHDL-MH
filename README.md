# Predict Hit Song from Spotify - Depend on Artist 

Dự án này sử dụng Python để phân tích dữ liệu khách hàng, từ đó tính toán các chỉ số marketing quan trọng và dự đoán hành vi mua hàng trong tương lai.

## 👥 Thành viên nhóm & Phân công công việc

Dự án được thực hiện bởi nhóm [Tên Nhóm], với sự đóng góp cụ thể của từng thành viên như sau:

| Họ và Tên | Vai trò | Chi tiết công việc thực hiện |
| :--- | :--- | :--- |
| **Đỗ Minh Huyền** | Trưởng nhóm / Data Analysist | Tìm kiếm dữ liệu, lên ý tưởng, và viết file `02_EDA.ipynb` để trực quan hóa dữ liệu gốc. |
| **Trần Huệ Chi** | Data Analysist | Viết file `02_EDA.ipynb`,để trực quan hóa dữ liệu gốc. |
| **Phạm Ngọc Yến Nhi** |Data Cleaner | Viết file `01_Loading & Cleaning.ipynb`, xử lý missing values, loại bỏ outliers và xuất ra file data chuẩn |
| **Nguyễn Huyền Trang** |Data Cleaner | Viết file `01_Loading & Cleaning.ipynb`,, xử lý missing values, loại bỏ outliers và xuất ra file data chuẩn |


## 📂 Cấu trúc Repository 


├── data/                           <- Thư mục chứa dữ liệu

├── raw_data.csv                <- Dữ liệu gốc tải về (Không chỉnh sửa)

└── cleaned_data.csv            <- Dữ liệu đã làm sạch (Dùng để chạy model)

├── notebooks/                      <- Thư mục chứa code Jupyter Notebook

│   ├── 01_KhamPhaDuLieu.ipynb      <- Bước 1: Khám phá và vẽ biểu đồ
│   ├── 02_LamSachDuLieu.ipynb      <- Bước 2: Xử lý dữ liệu lỗi
│   └── 03_HuanLuyenModel.ipynb     <- Bước 3: Chạy mô hình dự đoán
│
└── README.md                       <- Tài liệu hướng dẫn chung của dự án


## 🔍 Nguồn dữ liệu (Data Source)
* Dữ liệu được thu thập từ: [Chèn link hoặc tên nguồn dữ liệu vào đây]
* Mô tả ngắn: Tập dữ liệu gồm [số lượng] dòng và [số lượng] cột, chứa các thông tin như độ tuổi, thu nhập, lịch sử giao dịch...

## 💡 Kết quả nổi bật (Key Findings)
* **Về dữ liệu:** Phát hiện ra nhóm khách hàng có độ tuổi từ X đến Y mang lại tỷ lệ ROI cao nhất.
* **Về mô hình:** Thuật toán phân loại đã dự đoán chính xác tới [X]% khả năng quay lại của khách hàng.

## 🚀 Hướng dẫn chạy code (How to run)
Để tái tạo lại kết quả của nhóm, vui lòng thực hiện theo các bước sau:
1. Cài đặt các thư viện cần thiết: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
2. Mở Jupyter Notebook và chạy lần lượt các file trong thư mục `notebooks/` theo thứ tự từ `01` đến `03`.
