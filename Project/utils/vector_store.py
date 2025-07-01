import faiss
import numpy as np
import os
import json

class FaissStore:
    def __init__(self, dim, index_path="database/faiss.index", meta_path="database/faiss_meta.json"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path

        # Nếu đã có index lưu sẵn, load; nếu chưa, tạo mới
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            # IndexFlatL2 cho cosine/L2 (với normalize embedding trước)
            self.index = faiss.IndexFlatIP(dim)  # Inner Product cho cosine nếu bạn normalize embeddings
            self.metadata = {}  # id → user_id

    def add(self, user_id: str, embedding: np.ndarray):
        """
        Thêm một embedding mới:
        - user_id: chuỗi định danh user
        - embedding: numpy array kích thước (dim,)
        """
        idx = len(self.metadata)
        self.index.add(embedding.reshape(1, -1))
        self.metadata[str(idx)] = user_id
        self._save()

    def search(self, query_emb: np.ndarray, top_k: int = 5):
        """
        Tìm top_k embedding gần nhất:
        Trả về list [(user_id, score), ...]
        """
        D, I = self.index.search(query_emb.reshape(1, -1), top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1: 
                continue
            user_id = self.metadata.get(str(idx))
            results.append((user_id, float(dist)))
        return results

    def _save(self):
        """Ghi index và metadata ra file"""
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'w') as f:
            json.dump(self.metadata, f)
