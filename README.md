# Fuzzy C-means Clustering Algorithm (FCM)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Đây là dự án triển khai thuật toán **Fuzzy C-means (FCM)** để phân cụm dữ liệu bệnh tim mạch (Cardiovascular Disease Dataset). Dự án sử dụng thư viện `fcmeans` và thực hiện tiền xử lý dữ liệu, chuẩn hóa, giảm chiều (PCA) và đánh giá kết quả phân cụm.

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Dataset](#-dataset)
- [Thuật toán Fuzzy C-means](#-thuật-toán-fuzzy-c-means)
- [Cài đặt](#-cài-đặt)
- [Chạy dự án](#-chạy-dự-án)
- [Kết quả](#-kết-quả)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Phân tích chi tiết](#-phân-tích-chi-tiết)
- [Đóng góp](#-đóng-góp)
- [Giấy phép](#-giấy-phép)

---

## 🎯 Giới thiệu

**Fuzzy C-means (FCM)** là thuật toán phân cụm mờ (fuzzy clustering), cho phép một điểm dữ liệu thuộc về nhiều cụm khác nhau với các độ thuộc tính (membership degrees) khác nhau. Khác với K-means (hard clustering) - nơi mỗi điểm chỉ thuộc duy nhất một cụm, FCM gán độ thuộc thức [0, 1] cho mỗi điểm đối với từng cụm.

### Ứng dụng trong dự án
- Phân tích dữ liệu bệnh tim mạch (70,000 mẫu)
- Phân cụm bệnh nhân thành 2 nhóm: **có bệnh tim mạch** và **không bệnh tim mạch**
- So sánh kết quả phân cụm với nhãn thực tế (ground truth)

---

## 📊 Dataset

**Nguồn**: Cardiovascular Disease Dataset (Kaggle/UCI)

| Thông tin | Chi tiết |
|-----------|----------|
| **Số lượng mẫu** | 70,000 |
| **Số đặc trưng** | 12 (trước khi xử lý) |
| **Định dạng** | CSV, phân cách bằng `;` |
| **Nhãn mục tiêu** | `cardio` (0: không bệnh, 1: có bệnh) |

### Các đặc trưng (Features)

| Cột | Mô tả | Kiểu dữ liệu | Phạm vi |
|-----|-------|--------------|---------|
| `id` | ID bệnh nhân | int | 0 - 99,999 |
| `age` | Tuổi (ngày) | int | 10,798 - 23,713 |
| `gender` | Giới tính | int | 1 (nữ), 2 (nam) |
| `height` | Chiều cao | int | 55 - 250 cm |
| `weight` | Cân nặng | float | 10 - 200 kg |
| `ap_hi` | Huyết áp tâm thu | int | -150 - 16,020 |
| `ap_lo` | Huyết áp tâm trương | int | -70 - 11,000 |
| `cholesterol` | Cholesterol | int | 1 (bình thường), 2 (trên bình thường), 3 (cao) |
| `gluc` | Glucose | int | 1 (bình thường), 2 (trên bình thường), 3 (cao) |
| `smoke` | Hút thuốc | int | 0 (không), 1 (có) |
| `alco` | Uống rượu | int | 0 (không), 1 (có) |
| `active` | Hoạt động thể chất | int | 0 (không), 1 (có) |
| `cardio` | **Nhãn mục tiêu** | int | 0 (không bệnh), 1 (có bệnh) |

> **Lưu ý**: `age` được đo bằng ngày. Notebook đã chuyển đổi sang năm (`years = age / 365`) để dễ trực quan hóa.

---

## 🧮 Thuật toán Fuzzy C-means

### Nguyên lý toán học

FCM tối ưu hóa hàm mục tiêu sau:

$$J_m = \sum_{i=1}^{N} \sum_{j=1}^{C} u_{ij}^m \|x_i - c_j\|^2$$

Trong đó:
- $N$: số lượng điểm dữ liệu
- $C$: số cụm (clusters)
- $u_{ij}$: độ thuộc của điểm $x_i$ vào cụm $j$ ($0 \le u_{ij} \le 1$)
- $m$: hệ số mờ (fuzzifier), thường $m = 2$
- $c_j$: tâm của cụm $j$
- $\|x_i - c_j\|$: khoảng cách Euclid giữa điểm $x_i$ và tâm cụm $j$

### Các bước thuật toán

1. **Khởi tạo**: Chọn ngẫu nhiên ma trận độ thuộc $U^{(0)}$ hoặc tâm cụm $C^{(0)}$
2. **Cập nhật tâm cụm**:
   $$c_j = \frac{\sum_{i=1}^{N} u_{ij}^m x_i}{\sum_{i=1}^{N} u_{ij}^m}$$
3. **Cập nhật độ thuộc**:
   $$u_{ij} = \frac{1}{\sum_{k=1}^{C} \left( \frac{\|x_i - c_j\|}{\|x_i - c_k\|} \right)^{\frac{2}{m-1}}}$$
4. **Kiểm tra hội tụ**: Nếu $\|U^{(k+1)} - U^{(k)}\| < \epsilon$ thì dừng, ngược lại quay lại bước 2

### Ưu điểm so với K-means
- ✅ Xử lý tốt dữ liệu chồng chéo (overlapping clusters)
- ✅ Cho biết độ chắc chắn của việc phân cụm (membership degrees)
- ✅ Ít nhạy cảm với outliers nhờ cơ chế mờ
- ✅ Phù hợp cho y tế, sinh học - nơi ranh giới không rõ ràng

---

## ⚙️ Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- pip (package manager)

### Tạo môi trường ảo (khuyến nghị)

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Linux/Mac)
source venv/bin/activate
```

### Cài đặt dependencies

```bash
# Cập nhật pip
pip install --upgrade pip

# Cài đặt các thư viện cần thiết
pip install pandas numpy matplotlib seaborn scikit-learn fcmeans jupyter
```

Hoặc cài đặt từ `requirements.txt` (nếu có):

```bash
pip install -r requirements.txt
```

---

## 🚀 Chạy dự án

### Cách 1: Chạy Jupyter Notebook (Khuyến nghị)

```bash
# Khởi động Jupyter
jupyter notebook

# Mở file FCM.ipynb trong trình duyệt
# Chạy lần lượt các cell từ trên xuống dưới
```

### Cách 2: Chạy script Python (nếu chuyển đổi)

```bash
# Tạo file script từ notebook (tùy chọn)
jupyter nbconvert --to script FCM.ipynb

# Chạy
python FCM.py
```

### Quy trình thực thi trong Notebook

| Cell | Mô tả |
|------|-------|
| 0 | Import thư viện, đọc dữ liệu CSV |
| 1 | Thống kê mô tả dữ liệu (`describe()`) |
| 2 | Trực quan hóa phân bố tuổi theo nhãn cardio |
| 3 | Trực quan hóa các biến categorical |
| 4 | Kiểm tra giá trị thiếu (missing values) |
| 5 | Tách features (X) và labels (y) |
| 6 | Chuẩn hóa dữ liệu (StandardScaler) |
| 7 | Giảm chiều với PCA (2 components) |
| 8 | Khởi tạo và train FCM (n_clusters=2) |
| 9 | Dự đoán nhãn cụm |
| 10 | Trực quan hóa kết quả phân cụm trên không gian PCA |
| 11 | Tính độ chính xác (Accuracy) so với ground truth |
| 12-13 | Liệt kê các ID bị phân loại sai |

---

## 📈 Kết quả

### Hiệu suất mô hình

| Metric | Giá trị |
|--------|---------|
| **Accuracy** | **60.20%** |
| **Số cụm (n_clusters)** | 2 |
| **Số mẫu** | 70,000 |
| **Số chiều sau PCA** | 2 |

### Phân tích kết quả

- **Accuracy 60.2%**: Kết quả trung bình, cho thấy FCM đơn thuần chưa đủ tốt cho bài toán phân loại nhị phân này
- **Lý do có thể**:
  - FCM là unsupervised learning, không sử dụng nhãn `cardio` khi train
  - Dữ liệu có nhiều outliers (ap_hi max = 16020, ap_lo max = 11000)
  - Các đặc trưng categorical chưa được encode phù hợp
  - Cần feature engineering, lọc outliers, hoặc thử supervised learning

### Trực quan hóa

Notebook tạo ra 3 biểu đồ chính:
1. **Phân bố tuổi theo cardio** - Countplot tuổi vs nhãn bệnh
2. **Phân bố biến categorical** - Cholesterol, glucose, smoke, alco, active
3. **Kết quả FCM trên PCA 2D** - Scatter plot các cụm sau giảm chiều

---

## 📁 Cấu trúc dự án

```
FUZZY-C-MEANS---FCM/
├── FCM.ipynb                              # Notebook chính
├── cardiovascular_disease_dataset.csv     # Dữ liệu gốc (70k mẫu)
├── .gitignore                             # Git ignore rules
└── README.md                              # File này
```

---

## 🔍 Phân tích chi tiết & Gợi ý cải tiến

### Vấn đề hiện tại

1. **Outliers dữ liệu**: `ap_hi` (max 16020) và `ap_lo` (max 11000) có giá trị bất thường → cần lọc/IQR
2. **Chưa encode categorical**: `cholesterol`, `gluc` là ordinal nhưng chưa xử lý
3. **Chỉ dùng 2 cụm**: Có thể thử n_clusters=3,4,... với silhouette score
4. **Không so sánh với K-means**: Nên benchmark với K-means, Gaussian Mixture
5. **Chưa dùng membership degrees**: FCM cho ra độ thuộc, có thể dùng cho uncertainty quantification

### Gợi ý cải tiến

```python
# 1. Lọc outliers bằng IQR
Q1 = data['ap_hi'].quantile(0.25)
Q3 = data['ap_hi'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['ap_hi'] >= Q1 - 1.5*IQR) & (data['ap_hi'] <= Q3 + 1.5*IQR)]

# 2. Thử các số cụm khác nhau
from sklearn.metrics import silhouette_score
for k in range(2, 6):
    fcm = FCM(n_clusters=k)
    fcm.fit(X_scaled)
    labels = fcm.predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k={k}, Silhouette={score:.4f}")

# 3. Sử dụng membership matrix
u = fcm.u_  # Ma trận độ thuộc (n_samples, n_clusters)
# Phân tích uncertainty: entropy của membership
entropy = -np.sum(u * np.log(u + 1e-10), axis=1)
```

### Mở rộng

- [ ] Thêm **Elbow Method** / **Silhouette Analysis** để chọn k tối ưu
- [ ] So sánh với **K-means**, **Gaussian Mixture Model**, **DBSCAN**
- [ ] Thử **Supervised Learning** (Random Forest, XGBoost) để benchmark
- [ ] **Feature Engineering**: BMI, MAP (Mean Arterial Pressure), age groups
- [ ] **Cross-validation** cho đánh giá robust hơn
- [ ] Deploy thành **Streamlit/Gradio app** để demo tương tác

---

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Mở **Pull Request**

### Báo cáo lỗi / Yêu cầu tính năng

Mở [Issue](https://github.com/Vtl0303/FUZZY-C-MEANS---FCM/issues) mới với template:
- **Bug report**: Mô tả lỗi, bước tái hiện, expected behavior
- **Feature request**: Mô tả tính năng, use case, benefit

---

## 📄 Giấy phép

Dự án này được phân phối dưới giấy phép **MIT License**. Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

## 👨‍💻 Tác giả

**Vtl0303** - [GitHub Profile](https://github.com/Vtl0303)

---

## 🙏 Lời cảm ơn

- Thư viện [`fcmeans`](https://github.com/gaulinmp/fcmeans) cho implementation FCM
- Dataset **Cardiovascular Disease** từ Kaggle/UCI ML Repository
- Các thư viện: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

---

> ⭐ Nếu dự án hữu ích, hãy cho một **star** trên GitHub để ủng hộ!