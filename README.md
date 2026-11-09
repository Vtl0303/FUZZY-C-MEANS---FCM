# FUZZY-C-MEANS---FCM
dự đoán bệnh tim mạch sử dụng thuật toán phân cụm mờ 
# Thuật toán phân cụm mờ C-means (FCM)

[ ![ Python ] ( https://img.shields.io/badge/Python-3.8%2B-blue.svg )] ( https://www.python.org/downloads/ )
[ ![ Jupyter ] ( https://img.shields.io/badge/Jupyter-Notebook-orange.svg )] ( https://jupyter.org/ )
[ ![ Giấy phép ] ( https://img.shields.io/badge/License-MIT-green.svg )] ( GIẤY PHÉP )

Đây là dự án phát triển khai thuật toán ** Fuzzy C-means (FCM) ** để phân cụm dữ liệu bệnh tim mạch (Bộ dữ liệu bệnh tim mạch). Dự án sử dụng thư viện ` fcmeans ` và thực hiện tiền xử lý dữ liệu, chuẩn hóa, giảm chiều (PCA) và đánh giá kết quả phân cụm.

---

## 📋 Mục lục

-  [ Giới thiệu ] ( #-giới-thiệu )
-  [ Tập dữ liệu ] ( #-tập dữ liệu )
-  [ Thuật toán mờ C-means ] ( #-thuật-toán-fuzzy-c-means )
-  [ Cài đặt ] ( #-cài-đặt )
-  [ Chạy dự án ] ( #-chạy-dự-án )
-  [ Kết quả ] ( #-kết-quả )
-  [ Cấu trúc dự án ] ( #-cấu-dự-án )
-  [ Phân tích chi tiết ] ( #-phân-tích-chi-tiết )
-  [ Đóng góp ] ( #-đóng-góp )
  

---

## 🎯 Giới thiệu

** Fuzzy C-means (FCM) ** là thuật toán phân cụm mờ (phân cụm mờ), cho phép một dữ liệu thuộc về nhiều cụm khác nhau với các thuộc tính (cấp độ thành viên) khác nhau. Khác với K-mean (phân cụm cứng) - nơi mỗi điểm chỉ thuộc duy nhất một cụm, FCM phân bổ độ thuộc thức [ 0, 1 ] cho mỗi điểm đối với từng cụm.
** Thuật toán phân cụm mờ C-means được kế thừa và phát triển tư tưởng của thuật toán phân cụm rõ K-means. Cả hai thuật toán này đều sử dụng chung một chiến lược đó là phân cụm dữ liệu. Thuật toán phân cụm dữ liệu mờ C-means hay còn gọi tắt là thuật toán FCM (Fuzzy C-means) đã được áp dụng thành công trong việc giải quyết một số các bài toán như trong nhận dạng mẫu, xử lý ảnh, y học,...   

### Ứng dụng trong dự án
- Phân tích mạch dữ liệu (70.000 mẫu)
- Phân cụm bệnh nhân thành 2 nhóm: ** có bệnh tim mạch ** và ** không có bệnh tim mạch **
- So sánh kết quả phân cụm với nhãn thực tế

---

## 📊 Bộ dữ liệu

** Nguồn ** : Bộ dữ liệu bệnh tim mạch (Kaggle/UCI)

| Thông tin | Chi tiết |
| ----------- | ---------- |
|  ** Mẫu số lượng **  | 70.000 |
|  ** Số cụ thể **  | 12 (trước khi xử lý) |
|  ** Format **  | CSV, sử dụng ` ; `  |
|  ** Dán nhãn mục **  |  ` cardio` ( 0: không bệnh, 1: có bệnh) |

### Các cụ thể (Tính năng)

| Cột | Description | Data Type | Phạm vi |
| ----- | ------- | -------------- | --------- |
|  ` id` | ​ ID bệnh nhân | int | 0 - 99.999 |
|  ` tuổi `  | Tuổi (ngày) | int | 10.798 - 23.713 |
|  ` giới tính `  | Giới tính | int | 1 (nữ), 2 (nam) |
|  ` chiều cao `  | Chiều cao | int | 55 - 250 cm |
|  ` trọng lượng `  | Cân nặng | phao | 10 - 200 kg |
|  ` ap_hi` | ​ Áp dụng tâm thu | int | -150 - 16.020 |
|  ` ap_lo `  | Áp lực tâm lý | int | -70 - 11.000 |
|  ` cholesterol` | ​ Cholesterol | int | 1 (bình thường), 2 (bình thường), 3 (cao) |
|  ` gluc` | ​ Đường | int | 1 (bình thường), 2 (bình thường), 3 (cao) |
|  ` khói `  | Hút thuốc | int | 0 (không), 1 (có) |
|  ` rượu `  | Uống rượu | int | 0 (không), 1 (có) |
|  ` hoạt động `  | Hoạt động chất | int | 0 (không), 1 (có) |
|  ` tim mạch `  |  ** Dán nhãn mục **  | int | 0 (không bệnh), 1 (có bệnh) |

> ** Lưu ý ** : ` tuổi ` được đo bằng ngày. Notebook đã chuyển sang năm ( ` năm = tuổi / 365 ` ) để dễ dàng trực quan hóa.
---

## 🧮 Thuật toán Mờ C-means

### Nguyên lý thanh toán học

Mục tiêu tối ưu hóa FCM sau:

$$ J_m = \sum_{i=1}^{N} \sum_{j=1}^{C} u_{ij}^m \|x_i - c_j\|^2 $$

Trong đó:
- $N$: số lượng dữ liệu
- $C$: số cụm (clusters)
- $u _ {ij}$: độ thuộc tính của điểm $x_i$ vào cụm $j$ ($0 \le u _ {ij} \le 1$)
- $m$: hệ số mờ (fuzzifier), thường $m = 2$
- $c_j$: tâm của cụm $j$
- $ \| x_i - c_j \| $: khoảng cách Euclid giữa điểm $x_i$ và cụm $j$

### Các bước thuật toán

1 .  ** Khởi tạo ** : Chọn ngẫu nhiên ma trận thuộc tính $U^{(0)}$ hoặc tâm cụm $C^{(0)}$
2 .  ** Cập nhật tâm cụm ** :
   $$ c_j = \frac{\sum_{i=1}^{N} u_{ij}^m x_i}{\sum_{i=1}^{N} u_{ij}^m} $$
3 .  ** Cập nhật thuộc tính ** :
   $$ u_{ij} = \frac{1}{\sum_{k=1}^{C} \left( \frac{\|x_i - c_j\|}{\|x_i - c_k\|} \right)^{\frac{2}{m-1}}} $$
4 .  ** Kiểm tra hội tụ ** : If $ \| U^{(k+1)} - U^{(k)} \| < \epsilon$ thì dừng, lùi lại quay lại bước 2

### Ưu điểm với K-mean
- ✅ Xử lý chéo dữ liệu tốt (cụm chồng chéo)
- ✅ Biết mức độ chắc chắn của phân cụm công việc (bằng cấp thành viên)
- ✅ Giảm cảm giác với các ngoại lệ nhờ cơ chế làm mờ
- ✅ Phù hợp cho y tế, sinh học - nơi ranh giới chưa rõ rõ ràng

---

## ⚙️ Cài đặt

### Yêu cầu hệ thống
- Python 3.8 trở lên
- pip (trình quản lý gói)

### Tạo môi trường ảo (khuyến nghị)

``` bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Windows)
venv \ Scripts \ Activate

# Kích hoạt (Linux/Mac)
nguồn venv/bin/activate
```

### Cài đặt các phụ thuộc

``` bash
# đăng ký sớm pip
pip install --upgrade pip

# Cài đặt các thư viện cần thiết
pip install pandas numpy matplotlib seaborn scikit-learn fcmeans jupyter
```

Hoặc cài đặt từ ` requirements.txt ` (nếu có):

``` bash
pip install -r requirements.txt
```

---

## 🚀 Chạy dự án

### Cách 1: Chạy Jupyter Notebook (Khuyến nghị)

``` bash
# Khởi động Jupyter
sổ tay Jupyter

# Mở file FCM.ipynb trong trình duyệt
# Chạy lần lượt các ô từ trên xuống dưới
```

### Cách 2: Chạy script Python (nếu chuyển đổi)

``` bash
# Tạo tập lệnh từ sổ ghi chép (tùy chọn)
jupyter nbconvert --to script FCM.ipynb

# Chạy
python FCM.py
```

### Quy trình thực thi trong Notebook

| Tế bào | Mô tả |
| ------ | ------- |
| 0 | Nhập thư viện, đọc CSV dữ liệu |
| 1 | Thống kê dữ liệu mô tả ( ` mô tả() ` ) |
| 2 | Bố cục phân hóa trực tiếp theo nhãn tim |
| 3 | Trực tiếp hóa các biến thể loại |
| 4 | Kiểm tra việc thiếu giá trị (thiếu giá trị) |
| 5 | Tách các tính năng (X) và nhãn (y) |
| 6 | Chuẩn hóa dữ liệu (StandardScaler) |
| 7 | Giảm chiều với PCA (2 thành phần) |
| 8 | Khởi tạo và huấn luyện FCM (n_clusters=2) |
| 9 | Dự đoán cụm nhãn |
| 10 | Phân tích kết quả trực tiếp trên PCA không gian |
| 11 | Tính độ chính xác (Accuracy) so với sự thật |
| 13-12 | Liệt kê sai loại ID bị phân loại |

---

## 📈 Kết quả

### Hiệu mô hình

| Hệ mét | Giá trị |
| -------- | --------- |
|  ** Độ chính xác **  |  ** 60,20% **  |
|  ** Số cụm (n_clusters) **  | 2 |
|  ** Số mẫu **  | 70,000 |
|  ** Số chiều sau PCA **  | 2 |

### Phân tích kết quả

-  ** Độ chính xác 60,2% ** : Kết quả trung bình, cho thấy FCM đơn thuần chưa đủ tốt cho bài toán phân loại nhị phân này
-  ** Lý do có thể ** :
  - FCM là học tập không giám sát , không sử dụng nhãn ` cardio` khi tập luyện
  - Dữ liệu có nhiều ngoại lệ (ap_hi max = 16020, ap_lo max = 11000)
  - Các phân loại cụ thể chưa được mã hóa phù hợp
  - Cần có kỹ thuật tính năng, lọc các ngoại lệ hoặc thử nghiệm học tập có giám sát

### Trực quan

Notebook tạo ra 3 biểu đồ chính:
1 .  ** Phân bố tuổi theo cardio ** - Countplot tuổi vs nhãn bệnh
2 .  ** Phân bố biến thể loại ** - Cholesterol, glucose, khói, alco, hoạt tính
3 .  ** Kết quả FCM trên PCA 2D ** - Phân tán biểu đồ các cụm sau giảm chiều

---

## 📁 Cấu trúc dự án

```
FUZZY-C-MEANS---FCM/
├── FCM.ipynb # Notebook chính
├── cardio_disease_dataset.csv # Dữ liệu gốc (70k mẫu)
├── .gitignore # Các quy tắc bỏ qua của Git
└── README.md # File này
```

---

## 🔍 Phân tích chi tiết & Giảmi ý cải tiến

### Vấn đề hiện tại

1 .  ** Dữ liệu ngoại lệ ** : ` ap_hi ` (tối đa 16020) và ` ap_lo ` (tối đa 11000) có giá trị bất ngờ → cần lọc/IQR
2 .  ** Chưa mã hóa phân loại ** : ` cholesterol ` , ` gluc ` là thứ tự nhưng chưa được xử lý
3 .  ** Chỉ dùng 2 cụm ** : Có thể thử n_clusters=3,4,... với điểm hình bóng
4 .  ** Không so sánh với K-means ** : Nên benchmark với K-means, Gaussian Mixture
5 .  ** Chưa sử dụng cấp độ thành viên ** : FCM cho thuộc tính, có thể dùng để định lượng độ không chắc chắn

### Gợi ý cải tiến

``` python
# 1. Lọc các ngoại lệ bằng IQR
Q1 = data[ ' ap_hi ' ].quantile( 0.25 )
Q3 = dữ liệu [ ' ap_hi ' ].quantile( 0,75 )
IQR  = Q3 - Q1
data = data[(data[ ' ap_hi ' ] >= Q1 -  1.5 * IQR ) & (data[ ' ap_hi ' ] <= Q3 +  1.5 * IQR )]

# 2. Thử các cụm số khác nhau
from sklearn.metrics import silhouette_score
cho k trong  phạm vi ( 2 , 6 ):
    fcm = FCM( n_clusters = k)
    fcm.fit(X_scaled)
    nhãn = fcm.predict(X_scaled)
    điểm số = silhouette_score(X_scaled, labels)
    print ( f " k= { k } , Silhouette= { score :.4f } " )

# 3. Sử dụng ma trận thành viên
u = fcm.u_   # Ma trận thuộc tính (n_samples, n_clusters)
# Độ bất định của phân tích: entropy của thành viên
entropy =  - np.sum(u * np.log(u +  1e-10 ), axis = 1 )
```

### Mở rộng

-  [ ] Thêm ** Elbow Method ** / ** Silhouette Analysis ** để chọn k mức độ ưu tiên
-  [ ] So sánh với ** K-means ** , ** Mô hình hỗn hợp Gaussian ** , ** DBSCAN **
-  [ ] Thử ** Học có giám sát ** (Rừng ngẫu nhiên, XGBoost) để đánh giá hiệu năng
-  [ ]  ** Kỹ thuật đặc trưng ** : BMI, MAP (Huyết áp động mạch trung bình), nhóm tuổi
-  [ ]  ** Xác thực chéo ** cho đánh giá mạnh mẽ hơn
-  [ ] Triển khai thành ** Streamlit/Gradio app ** để demo tương tác

---

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh! Xin vui lòng:

1. Sao chép kho lưu trữ
2 . Tạo nhánh mới ( ` git register -b feature/AmazingFeature ` )
3. Cam kết thay đổi ( ` git commit -m 'Add some AmazingFeature' ` )
4. Đẩy nhánh lên ( ` git push origin feature/ AmazingFeature ` )
5. Mở ** Yêu cầu kéo **

### Báo cáo lỗi / Yêu cầu tính năng

Mở [ Issue ] ( https://github.com/Vtl0303/FUZZY-C-MEANS---FCM/issues ) mới với mẫu:
-  ** Báo cáo lỗi ** : Mô tả lỗi, bước tái hiện, hành vi dự kiến
-  ** Yêu cầu tính năng ** : Mô tả tính năng, use case, lợi ích


## Tác giả

** Vtl0303 ** - [ Hồ sơ GitHub ] ( https://github.com/Vtl0303 )

---

##  Lời cảm ơn

- Thư viện [ ` fcmeans ` ] ( https://github.com/gaulinmp/fcmeans ) để triển khai FCM
- Bộ dữ liệu ** Bệnh tim mạch ** từ Kaggle/UCI ML Repository
- Các thư viện : ` scikit - learn` , ` pandas` , ` numpy` , ` matplotlib ` , ` seaborn`

