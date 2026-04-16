# Báo Cáo Kỹ Thuật: Triển Khai Thuật Toán Hierarchical Clustering

> **Tài liệu tham khảo:** [GeeksforGeeks - Hierarchical Clustering](https://www.geeksforgeeks.org/machine-learning/hierarchical-clustering/)
> **Mục tiêu:** Hiểu sâu về cơ chế phân cụm phân cấp (Agglomerative), cách đọc Dendrogram và lựa chọn Linkage phù hợp.

---

## 🛠️ PHẦN 1: TỪNG BƯỚC IMPLEMENT CODE (STEP-BY-STEP)

Khác với K-Means, Hierarchical Clustering (Phân cụm phân cấp) không yêu cầu khai báo số cụm $k$ ngay từ đầu. Chúng ta sẽ tập trung vào phương pháp **Agglomerative** (từ dưới lên).

### 1️⃣ Bước 1: Import các thư viện
Chúng ta cần `scipy.cluster.hierarchy` để vẽ Dendrogram và `sklearn.cluster` để thực hiện phân cụm.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
```

### 2️⃣ Bước 2: Chuẩn bị dữ liệu
Tạo tập dữ liệu giả lập với các cụm tách biệt để quan sát quá trình gộp nhóm.

```python
# Tạo dữ liệu mẫu
X, y = make_blobs(n_samples=200, centers=4, cluster_std=0.60, random_state=0)

plt.figure(figsize=(8, 5))
plt.scatter(X[:,0], X[:,1], s=30)
plt.title("Dữ liệu thô ban đầu")
plt.show()
```

### 3️⃣ Bước 3: Vẽ Dendrogram (Sơ đồ hình cây)
Đây là bước quan trọng nhất để xác định số lượng cụm tối ưu. Dendrogram ghi lại lịch sử của mọi lần gộp các điểm dữ liệu.



```python
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")

# Sử dụng phương pháp 'ward' để tối thiểu hóa phương sai trong cụm
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

plt.xlabel('Điểm dữ liệu')
plt.ylabel('Khoảng cách Euclidean')
plt.show()
```

### 4️⃣ Bước 4: Huấn luyện mô hình Agglomerative
Dựa vào Dendrogram, nếu ta cắt ở mức khoảng cách lớn nhất mà không đi ngang qua nhiều đường dọc, ta chọn được số cụm (ví dụ $n\_clusters=4$).

```python
# Khởi tạo mô hình
hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')

# Dự đoán cụm
y_hc = hc.fit_predict(X)
```

### 5️⃣ Bước 5: Trực quan hóa kết quả
Tô màu các điểm dựa trên cụm mà chúng được gán vào.

```python
plt.figure(figsize=(8, 5))
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=50, c='red', label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=50, c='green', label='Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=50, c='cyan', label='Cluster 4')

plt.title('Kết quả phân cụm Hierarchical')
plt.legend()
plt.show()
```

---

## 🧠 PHẦN 2: KẾT LUẬN & PHÂN TÍCH CHUYÊN SÂU

Dưới góc nhìn của một ML Engineer, thuật toán Hierarchical Clustering có những đặc điểm cực kỳ khác biệt so với K-Means:

### 1. Phân tích về Linkage (Tiêu chí liên kết) 🔗
Việc chọn `linkage` thay đổi hoàn toàn "hình dáng" của cụm:
* **Ward:** Thường là lựa chọn mặc định tốt nhất vì nó giảm thiểu phương sai trong cụm (giống K-Means), tạo ra các cụm có kích thước đồng đều.
* **Single Linkage (Min):** Dễ bị hiện tượng "chaining" (các cụm bị kéo dài và dính vào nhau bởi một vài điểm nhiễu).
* **Complete Linkage (Max):** Ưu tiên các cụm có đường kính nhỏ, tạo ra các nhóm rất chặt chẽ.

### 2. Ưu điểm vượt trội về Cấu trúc Phân cấp
Hierarchical Clustering không chỉ phân cụm mà còn cho ta biết **mối quan hệ** giữa các cụm. 
* **Ứng dụng:** Rất mạnh trong tin sinh học (xây dựng cây phát sinh loài) hoặc phân loại sản phẩm trong thương mại điện tử (Category -> Sub-category).

### 3. Hạn chế về Hiệu năng (Scalability) 📉
Đây là "gót chân Achilles" của thuật toán này:
* **Độ phức tạp thời gian:** Thường là $O(N^3)$ hoặc $O(N^2 \log N)$. 
* **Vấn đề thực tế:** Với tập dữ liệu hàng triệu dòng, Hierarchical Clustering sẽ cực kỳ chậm và tốn RAM để lưu trữ ma trận khoảng cách. K-Means (với độ phức tạp tuyến tính) sẽ thắng thế trong trường hợp này.

### 4. Tính bất biến (Irreversibility)
Một khi hai điểm đã được gộp vào một cụm ở bước dưới, chúng sẽ **không bao giờ** bị tách ra ở các bước trên. Điều này có nghĩa là nếu một bước gộp nhóm bị sai do nhiễu, sai lầm đó sẽ kéo theo kết quả của toàn bộ cây.