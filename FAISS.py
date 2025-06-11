import torch
import faiss
import numpy as np
# Giả sử đã có embedder và danh sách ảnh face_imgs
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class FaissVectorDB:
    def __init__(self, dim=512, threshold=0.7):
        self.index = faiss.IndexFlatIP(dim)  # Cosine similarity vì vector đã chuẩn hóa
        self.vectors = []  # Để mapping kết quả về ID hoặc dữ liệu gốc
        self.threshold = threshold

    def add(self, embeddings: torch.Tensor, metadata: list = None):
        """
        embeddings: torch.Tensor có shape (N, 512), đã được chuẩn hóa
        metadata: danh sách thông tin tương ứng (vd: tên, ID)
        """
        vecs = embeddings.cpu().numpy().astype('float32')
        self.index.add(vecs)
        self.vectors.extend(metadata if metadata else [None] * len(vecs))

    def search(self, query: torch.Tensor, topk=1):
        """
        query: torch.Tensor shape (1, 512), đã chuẩn hóa
        Trả về danh sách tuple (score, metadata)
        """
        q = query.cpu().numpy().astype('float32')
        scores, indices = self.index.search(q, topk)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= self.threshold:
                results.append((score, self.vectors[idx]))
        return results
