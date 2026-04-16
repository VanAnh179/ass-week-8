# Báo Cáo Kỹ Thuật: Phân Khúc Khách Hàng (Customer Segmentation)

> **Tài liệu tham khảo:** [GeeksforGeeks - Customer Segmentation using Unsupervised ML](https://www.geeksforgeeks.org/machine-learning/customer-segmentation-using-unsupervised-machine-learning-in-python/)
> **Mục tiêu:** Áp dụng học máy không giám sát để chuyển đổi dữ liệu thô thành các nhóm khách hàng có giá trị kinh doanh (Actionable Insights).

---

## 🛠️ PHẦN 1: TỪNG BƯỚC IMPLEMENT CODE (END-TO-END)

Chúng ta sẽ sử dụng bộ dữ liệu kinh điển **Mall Customers Dataset** để phân loại khách hàng dựa trên thu nhập và hành vi chi tiêu.

### 1️⃣ Bước 1: Khám phá dữ liệu (EDA)
Trước khi phân cụm, chúng ta cần hiểu các phân phối cơ bản.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dữ liệu
df = pd.read_csv('Mall_Customers.csv')

# Kiểm tra phân phối thu nhập và điểm chi tiêu
plt.figure(figsize=(10, 5))
sns.histplot(df['Annual Income (k$)'], kde=True, color='blue')
plt.title('Phân phối thu nhập hàng năm')
plt.show()
```

### 2️⃣ Bước 2: Lựa chọn đặc trưng (Feature Selection)
Để trực quan hóa tốt nhất trên không gian 2D, chúng ta tập trung vào hai biến quan trọng nhất: **Annual Income** (Thu nhập) và **Spending Score** (Điểm chi tiêu).

```python
# Lấy cột thứ 3 (Income) và thứ 4 (Spending Score)
X = df.iloc[:, [3, 4]].values
```

### 3️⃣ Bước 3: Tìm số cụm tối ưu (Elbow Method)
Chúng ta tính toán **WCSS** (Within-Cluster Sum of Squares) cho các giá trị $k$ từ 1 đến 10.



```python
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Phương pháp khuỷu tay (Elbow Method)')
plt.xlabel('Số lượng cụm (k)')
plt.ylabel('WCSS')
plt.show()
```

### 4️⃣ Bước 4: Huấn luyện mô hình với $k=5$
Dựa trên biểu đồ Elbow, điểm "gãy" rõ nhất là tại $k=5$.

```python
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
```

### 5️⃣ Bước 5: Trực quan hóa các phân khúc
Đây là lúc chúng ta gán nhãn cho từng nhóm khách hàng.

```python
plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
labels = ['Tiết kiệm', 'Trung bình', 'Mục tiêu', 'Vung tay quá trán', 'Cẩn trọng']

for i in range(5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i], label=labels[i])

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Phân khúc khách hàng trung tâm thương mại')
plt.xlabel('Thu nhập hàng năm (k$)')
plt.ylabel('Điểm chi tiêu (1-100)')
plt.legend()
plt.show()
```

---

## 🧠 PHẦN 2: KẾT LUẬN & PHÂN TÍCH CHUYÊN SÂU

Dưới góc độ kỹ thuật, dự án này không chỉ là về code mà là về việc chuyển đổi dữ liệu thành chiến lược.

### 1. Phân tích 5 nhóm khách hàng 📊
Dựa vào biểu đồ kết quả, chúng ta có 5 hồ sơ khách hàng rõ rệt:
1.  **Cụm Mục Tiêu (Target):** Thu nhập cao + Chi tiêu cao. Đây là nhóm cần ưu tiên chăm sóc khách hàng VIP.
2.  **Cụm Vung tay (Spendthrifts):** Thu nhập thấp + Chi tiêu cao. Nhóm này phản ứng tốt với các chương trình khuyến mãi ngắn hạn.
3.  **Cụm Cẩn trọng (Careful):** Thu nhập cao + Chi tiêu thấp. Cần các chiến dịch marketing cá nhân hóa để kích cầu.
4.  **Cụm Tiết kiệm (Sensible):** Thu nhập thấp + Chi tiêu thấp. Khách hàng nhạy cảm về giá.
5.  **Cụm Trung bình (Standard):** Mọi thứ ở mức vừa phải.

### 2. Lời khuyên của ML Engineer về "Feature Engineering" 🛠️
Trong bài toán thực tế, chúng ta không nên chỉ dùng 2 biến. Tuy nhiên, việc sử dụng nhiều biến (như Tuổi, Giới tính, Tần suất mua hàng) sẽ dẫn đến **Lời nguyền đa chiều (Curse of Dimensionality)**. 
* **Giải pháp:** Sử dụng **PCA (Principal Component Analysis)** để giảm chiều dữ liệu xuống 2D hoặc 3D trước khi chạy K-Means. Điều này giúp giữ lại thông tin quan trọng nhất mà vẫn đảm bảo thuật toán hoạt động hiệu quả.

### 3. Đánh giá chất lượng phân cụm 📏
Ngoài Elbow Method, một kỹ sư ML thực thụ sẽ sử dụng thêm **Silhouette Score**.
* Nếu Silhouette Score gần bằng 1: Các cụm được tách biệt rõ ràng.
* Nếu Silhouette Score gần bằng 0: Các cụm bị chồng lấn (Overlapping).
* Trong bài toán này, $k=5$ thường cho Silhouette Score tối ưu nhất trên Mall Dataset.

### 4. Giá trị thực tiễn (Business Value)
Phân cụm là bước đệm cho **Hệ thống gợi ý (Recommendation Systems)**. Thay vì gửi email quảng cáo giống nhau cho tất cả mọi người (Mass Marketing), doanh nghiệp có thể thực hiện **Targeted Marketing**, giúp tăng tỷ lệ chuyển đổi (Conversion Rate) và tối ưu hóa ngân sách quảng cáo.