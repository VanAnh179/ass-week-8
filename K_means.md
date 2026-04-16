# Báo Cáo Kỹ Thuật: Triển Khai Thuật Toán K-Means Clustering

> **Tài liệu tham khảo:** [GeeksforGeeks - K-Means Clustering Introduction](http://geeksforgeeks.org/machine-learning/k-means-clustering-introduction/)
> **Mục tiêu:** Nắm bắt cơ chế hoạt động cốt lõi của K-Means thông qua việc tự triển khai (implement) từng bước bằng Python và phân tích các ưu/nhược điểm trong thực tế.

---

## 🛠️ PHẦN 1: TỪNG BƯỚC IMPLEMENT CODE (STEP-BY-STEP)

Dưới đây là các bước xây dựng thuật toán K-Means từ con số không (from scratch), dựa trên hướng dẫn của bài viết.

### 1️⃣ Bước 1: Import các thư viện cần thiết
Chúng ta sử dụng `numpy` cho các phép toán ma trận, `matplotlib` để trực quan hóa và `make_blobs` từ `sklearn` để tạo dữ liệu giả lập.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
```

### 2️⃣ Bước 2: Tạo Custom Dataset (Dữ liệu nhân tạo)
Sinh ra 500 điểm dữ liệu trong không gian 2D, được chia sẵn thành 3 cụm (clusters) để dễ dàng kiểm chứng thuật toán.

```python
# Khởi tạo dữ liệu
X, y = make_blobs(n_samples=500, n_features=2, centers=3, random_state=23)

# Trực quan hóa dữ liệu thô
fig = plt.figure(0)
plt.grid(True)
plt.scatter(X[:,0], X[:,1])
plt.show()
```

### 3️⃣ Bước 3: Khởi tạo ngẫu nhiên Centroids (Tâm cụm)
Khởi tạo ngẫu nhiên vị trí cho `k=3` tâm cụm. Cấu trúc dictionary `clusters` được sử dụng để lưu trữ tọa độ tâm (`center`) và danh sách các điểm thuộc về cụm đó (`points`).

```python
k = 3
clusters = {}
np.random.seed(23) # Cố định seed để dễ tái lập kết quả

for idx in range(k):
    # Khởi tạo tọa độ tâm ngẫu nhiên
    center = 2 * (2 * np.random.random((X.shape[1],)) - 1)
    cluster = {
        'center': center,
        'points': []
    }
    clusters[idx] = cluster
```

### 4️⃣ Bước 4: Trực quan hóa Dữ liệu và Tâm cụm ban đầu
Hiển thị các tâm cụm vừa được khởi tạo ngẫu nhiên (chữ thập/sao màu đỏ) lên trên biểu đồ dữ liệu.

```python
plt.scatter(X[:,0], X[:,1])
plt.grid(True)

for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0], center[1], marker='*', c='red', s=150)
plt.show()
```

### 5️⃣ Bước 5: Định nghĩa hàm tính khoảng cách (Euclidean Distance)
K-Means phân nhóm dựa trên mức độ "gần gũi", và thước đo tiêu chuẩn ở đây là khoảng cách Euclidean.

```python
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))
```

### 6️⃣ Bước 6: Xây dựng hàm Gán cụm (Assign) & Cập nhật tâm (Update)
Đây là "trái tim" của thuật toán.
* **Hàm `assign_clusters`:** Duyệt qua mọi điểm dữ liệu, tính khoảng cách đến từng tâm và gán điểm đó vào tâm gần nhất.
* **Hàm `update_clusters`:** Tính toán lại tọa độ tâm cụm bằng cách lấy trung bình cộng (mean) tọa độ của tất cả các điểm đã được gán vào cụm đó.

```python
def assign_clusters(X, clusters):
    for idx in range(X.shape[0]):
        dist = []
        curr_x = X[idx]
        for i in range(k):
            dis = distance(curr_x, clusters[i]['center'])
            dist.append(dis)
        
        # Tìm tâm gần nhất và gán điểm vào cụm đó
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
    return clusters

def update_clusters(X, clusters):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            # Cập nhật tâm mới là giá trị trung bình của các điểm trong cụm
            new_center = points.mean(axis=0)
            clusters[i]['center'] = new_center
            clusters[i]['points'] = [] # Reset points cho vòng lặp tiếp theo
    return clusters
```

### 7️⃣ Bước 7 & 8: Chạy dự đoán và Thực thi
Hàm `pred_cluster` dùng để lấy ra label cụm của từng điểm. Sau đó gọi lần lượt các bước Assign và Update.

```python
def pred_cluster(X, clusters):
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i], clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred

# THỰC THI (1 bước lặp)
clusters = assign_clusters(X, clusters)
clusters = update_clusters(X, clusters)
pred = pred_cluster(X, clusters)
```

### 8️⃣ Bước 9: Trực quan hóa kết quả phân cụm
Vẽ lại biểu đồ với các điểm dữ liệu được tô màu theo cụm dự đoán và vị trí các tâm cụm mới.

```python
plt.scatter(X[:,0], X[:,1], c=pred, cmap='viridis')

for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0], center[1], marker='^', c='red', s=150)
plt.show()
```

---

## 🧠 PHẦN 2: KẾT LUẬN & PHÂN TÍCH CHUYÊN SÂU (GÓC NHÌN TỪ ML ENGINEER)

> 💡 **Nhận định chung:** Cách tiếp cận trong bài viết của GFG rất trực quan và tuyệt vời để học viên hiểu được **bản chất toán học** bên dưới lớp vỏ bọc của thư viện `sklearn`. Tuy nhiên, nếu mang đoạn code này áp dụng vào các bài toán thực tế (Production), nó vẫn còn nhiều "lỗ hổng" kỹ thuật.

Dưới vai trò là một người làm chuyên ngành Machine Learning, tôi có các phân tích và đề xuất cải tiến như sau:

### 1. Thiếu vòng lặp hội tụ (Convergence Loop) ⚠️
* **Vấn đề:** Trong bài viết (Bước 8), tác giả chỉ gọi hàm `assign_clusters` và `update_clusters` **đúng 1 lần**. Trong thực tế, K-Means là một thuật toán lặp (iterative algorithm). Tâm cụm cần được cập nhật liên tục cho đến khi hội tụ (tâm cụm không thay đổi vị trí nữa) hoặc đạt số lần lặp tối đa (max iterations).
* **Giải pháp:** Cần bọc Bước 8 trong một vòng lặp `while` hoặc `for`.
    ```python
    # Giả mã cải tiến
    for _ in range(max_iter):
        old_centers = get_current_centers(clusters)
        clusters = assign_clusters(X, clusters)
        clusters = update_clusters(X, clusters)
        if check_convergence(old_centers, new_centers):
            break
    ```

### 2. Vấn đề "Bẫy khởi tạo" (Initialization Trap) 🎯
* **Phân tích:** Việc sử dụng `np.random.random` (Bước 3) để chọn ngẫu nhiên vị trí tâm ban đầu là rất rủi ro. Nếu các tâm cụm bị khởi tạo quá gần nhau hoặc nằm lọt thỏm vào các vùng nhiễu, thuật toán sẽ bị mắc kẹt ở **cực tiểu cục bộ (local optima)**, dẫn đến kết quả phân cụm hoàn toàn sai lệch.
* **Giải pháp thực tế:** Trong Machine Learning hiện đại, chúng ta luôn ưu tiên sử dụng thuật toán **K-Means++** để khởi tạo tâm. K-Means++ đảm bảo các tâm ban đầu được đặt càng xa nhau càng tốt, giúp tăng tốc độ hội tụ và độ chính xác. Thư viện `sklearn.cluster.KMeans` đã mặc định sử dụng `init='k-means++'`.

### 3. Sự phụ thuộc vào khoảng cách (Scale Sensitivity) 📏
* **Phân tích:** Thuật toán K-Means bị chi phối hoàn toàn bởi hàm `Euclidean distance`. Nếu tập dữ liệu có các features ở các thang đo (scale) khác nhau (Ví dụ: Feature A dao động từ 1-10, Feature B dao động từ 1000-100000), thuật toán sẽ bị thiên lệch hoàn toàn về Feature B.
* **Best Practice:** TÔN CHỈ trong Machine Learning: **Luôn luôn Scale dữ liệu** (ví dụ dùng `StandardScaler` hoặc `MinMaxScaler`) trước khi đưa vào mô hình K-Means. Bài viết GFG dùng tập `make_blobs` có scale tương đương nhau nên không bộc lộ điểm yếu này.

### 4. Giả định về mặt hình học (Non-spherical Clusters) 🧩
* **Phân tích:** K-Means chỉ hoạt động tốt khi các cụm có hình dạng **hình cầu (spherical)** và mật độ tương đồng nhau. Nếu dữ liệu thực tế có dạng trăng khuyết (moons), hình vòng tròn lồng nhau, hoặc các cụm có mật độ dày mỏng khác nhau, K-Means sẽ thất bại thảm hại.
* **Giải pháp thay thế:** Nếu phân tích EDA (Exploratory Data Analysis) cho thấy dữ liệu có cấu trúc không gian phức tạp, thay vì ép dùng K-Means, ta nên chuyển sang các thuật toán phân cụm theo mật độ như **DBSCAN** hoặc mô hình xác suất như **Gaussian Mixture Models (GMM)**.

### 5. Bài toán chọn số `k`
* Bài viết có nhắc đến việc chọn số $k$ là một thách thức. Trong thực tế làm việc, chúng tôi hiếm khi "đoán" số $k$. Thay vào đó, chúng tôi chạy một vòng lặp thử nhiều giá trị $k$ và sử dụng các phương pháp định lượng như **Elbow Method (WCSS)** kết hợp với **Silhouette Score** để tìm ra số $k$ tối ưu nhất dựa trên dữ liệu.