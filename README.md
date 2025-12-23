# HPG Demo: Tripartite Heterogeneous Graph Propagation

Demo toy cho mô hình HGP từ bài báo "Tripartite Heterogeneous Graph Propagation for Large-scale Social Recommendation" (arXiv:1908.02569v1).

## Cài đặt
- Cài thư viện: `pip install -r requirements.txt`

## Chạy
- `python main.py`
- Kết quả: Huấn luyện 10 epochs, dự đoán CTR mẫu.

## Giải thích
- Dựa trên graph tripartite (Fig. 1).
- Propagation tránh oversmoothing (sec. 3, eq. 1-2).
- Attention kết hợp embeddings (sec. 4, eq. 3-4).
- CTR prediction (eq. 5).
- Để scale lớn: Thêm sampling (sec. 4.2).

Liên hệ: Nếu cần mở rộng với dữ liệu thực (BERT/VGG như sec. 5.1).