# Báo Cáo Kỹ Thuật: Thuật Toán Expectation-Maximization (EM)

> **Tài liệu tham khảo:** [GeeksforGeeks - EM Algorithm](https://www.geeksforgeeks.org/machine-learning/ml-expectation-maximization-algorithm/)
> **Mục tiêu:** Giải quyết bài toán ước lượng tham số trong mô hình có biến ẩn (Latent Variables), trọng tâm là Gaussian Mixture Models (GMM).

---

## 🛠️ PHẦN 1: TỪNG BƯỚC IMPLEMENT CODE (PHƯƠNG PHÁP GMM)

Thuật toán EM thường được minh họa rõ nhất qua **Gaussian Mixture Model (GMM)**. Đây là một dạng "phân cụm mềm" (soft clustering), nơi mỗi điểm dữ liệu thuộc về một cụm với một xác suất nhất định, thay vì chỉ thuộc về một cụm duy nhất như K-Means.

### 1️⃣ Bước 1: Khởi tạo tham số
Chúng ta cần khởi tạo các tham số cho $k$ phân phối chuẩn:
* **Mean ($\mu$):** Trung bình của mỗi cụm.
* **Variance ($\sigma^2$):** Độ phân tán của mỗi cụm.
* **Weights ($\pi$):** Tỷ trọng (xác suất) của mỗi cụm trong tổng thể.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Tạo dữ liệu mẫu từ 2 phân phối chuẩn khác nhau
data = np.concatenate([np.random.normal(0, 1, 300), np.random.normal(5, 1, 700)])

# Khởi tạo tham số ban đầu (giả định)
mu1, mu2 = -1, 1
sigma1, sigma2 = 1, 1
w1, w2 = 0.5, 0.5
```

### 2️⃣ Bước 2: E-Step (Expectation Step)
Trong bước này, chúng ta tính toán xác suất (trách nhiệm - responsibility) mà mỗi điểm dữ liệu $x_i$ thuộc về cụm $j$ dựa trên các tham số hiện tại.

$$b_{ij} = \frac{w_j \cdot \mathcal{N}(x_i | \mu_j, \sigma_j)}{\sum_{k} w_k \cdot \mathcal{N}(x_i | \mu_k, \sigma_k)}$$

```python
def e_step(data, mu1, mu2, sigma1, sigma2, w1, w2):
    # Tính xác suất mật độ (Likelihood)
    prob1 = w1 * norm.pdf(data, mu1, sigma1)
    prob2 = w2 * norm.pdf(data, mu2, sigma2)
    
    # Chuẩn hóa để tổng xác suất bằng 1 (Responsibilities)
    total_prob = prob1 + prob2
    r1 = prob1 / total_prob
    r2 = prob2 / total_prob
    return r1, r2
```

### 3️⃣ Bước 3: M-Step (Maximization Step)
Sử dụng các giá trị "trách nhiệm" vừa tính được để cập nhật lại các tham số $\mu, \sigma, w$ nhằm tối đa hóa hàm Likelihood.

```python
def m_step(data, r1, r2):
    # Cập nhật Weights
    w1 = np.mean(r1)
    w2 = np.mean(r2)
    
    # Cập nhật Means (Weighted Average)
    mu1 = np.sum(r1 * data) / np.sum(r1)
    mu2 = np.sum(r2 * data) / np.sum(r2)
    
    # Cập nhật Variances
    sigma1 = np.sqrt(np.sum(r1 * (data - mu1)**2) / np.sum(r1))
    sigma2 = np.sqrt(np.sum(r2 * (data - mu2)**2) / np.sum(r2))
    
    return mu1, mu2, sigma1, sigma2, w1, w2
```

### 4️⃣ Bước 4: Vòng lặp hội tụ
Lặp lại E-step và M-step cho đến khi các tham số không còn thay đổi đáng kể.

```python
for i in range(100):
    r1, r2 = e_step(data, mu1, mu2, sigma1, sigma2, w1, w2)
    mu1, mu2, sigma1, sigma2, w1, w2 = m_step(data, r1, r2)
    
    if i % 20 == 0:
        print(f"Iteration {i}: mu1={mu1:.2f}, mu2={mu2:.2f}")
```

---

## 🧠 PHẦN 2: KẾT LUẬN & PHÂN TÍCH CHUYÊN SÂU



Dưới góc nhìn chuyên môn, thuật toán EM là một khung lý thuyết mạnh mẽ hơn nhiều so với các thuật toán phân cụm đơn thuần.

### 1. K-Means vs EM: Sự khác biệt về triết lý
* **K-Means (Hard Assignment):** Một điểm dữ liệu chỉ có thể thuộc 1 cụm. Điều này gây sai số lớn nếu dữ liệu nằm ở vùng giao thoa.
* **EM/GMM (Soft Assignment):** Một điểm thuộc về cụm A với 70% xác suất và cụm B với 30%. Điều này phản ánh thực tế dữ liệu tự nhiên tốt hơn, đặc biệt trong phân khúc khách hàng hoặc xử lý hình ảnh.

### 2. Sức mạnh của Biến ẩn (Latent Variables) 👻
Trong thực tế, chúng ta thường chỉ quan sát được kết quả ($x$) mà không biết nguyên nhân hoặc nhóm gốc ($z$). EM cho phép chúng ta "đoán" $z$ (E-step) rồi dùng dự đoán đó để tinh chỉnh mô hình (M-step). Đây là nền tảng cho nhiều mô hình phức tạp như **Hidden Markov Models (HMM)** trong nhận dạng giọng nói.

### 3. Nhược điểm cần lưu ý (Engineer's Warning) ⚠️
* **Cực tiểu cục bộ (Local Optima):** Giống K-Means, EM rất nhạy cảm với việc khởi tạo tham số ban đầu. Nếu khởi tạo sai, nó có thể hội tụ về một kết quả không tối ưu.
* **Độ phức tạp tính toán:** Mỗi vòng lặp của EM tốn kém hơn K-Means vì phải tính toán hàm mật độ xác suất (PDF) cho mọi điểm dữ liệu đối với mọi cụm.
* **Vấn đề Singularities:** Nếu một cụm chỉ chứa 1 điểm dữ liệu, Variance có thể tiến về 0, làm cho Likelihood tiến tới vô hạn và làm sập thuật toán.

### 4. Kết luận
Thuật toán EM là công cụ "phải biết" đối với ML Engineer khi đối mặt với dữ liệu bị thiếu (missing data) hoặc các mô hình xác suất phức tạp. Nó không chỉ là thuật toán phân cụm, mà là một phương pháp tối ưu hóa tổng quát cho các bài toán ước lượng tham số.