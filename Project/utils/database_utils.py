import json
import os
import numpy as np

# Paths setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DB_DIR = os.path.join(BASE_DIR, 'database')
USERS_JSON = os.path.join(DB_DIR, 'user_db.json')
IMAGES_DIR = os.path.join(DB_DIR, 'images')
EMBEDDINGS_DIR = os.path.join(DB_DIR, 'embeddings')

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
if not os.path.exists(USERS_JSON):
    with open(USERS_JSON, 'w', encoding='utf-8') as f:
        json.dump([], f)


def load_database():
    with open(USERS_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)
def save_database(db):
    with open(USERS_JSON, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def add_user(name, age, major, course, gmail, phone, embedding):
    """
    Thêm user mới, lưu embedding và metadata
    """
    db = load_database()
    user_id = len(db) + 1

    # Lưu embedding dưới dạng .npy
    emb_path = os.path.join(EMBEDDINGS_DIR, f'user_{user_id}.npy')
    np.save(emb_path, embedding)

    # Tạo record metadata
    record = {
        'id': user_id,
        'name': name,
        'age': age,
        'major': major,
        'course': course,
        'gmail': gmail,
        'phone': phone,
        'image': os.path.relpath(os.path.join(IMAGES_DIR, f'user_{user_id}.jpg'), BASE_DIR),
        'embedding': os.path.relpath(emb_path, BASE_DIR)
    }
    db.append(record)
    save_database(db)
    return user_id


def find_user_by_embedding(embedding, threshold=0.6):
    """
    So khớp embedding với cơ sở dữ liệu, trả về record và score
    """
    db = load_database()
    best_match, best_score = None, threshold
    for rec in db:
        db_emb = np.load(os.path.join(BASE_DIR, rec['embedding']))
        score = np.dot(db_emb, embedding) / (np.linalg.norm(db_emb) * np.linalg.norm(embedding))
        print(score)
        if score > best_score:
            best_score, best_match = score, rec
    return best_match, best_score

def get_user_info(user_id):
    """
    Trả về record user (dict) từ user_db.json theo id,
    hoặc None nếu không tìm thấy.
    """
    db = load_database()
    try:
        uid = int(user_id)
    except ValueError:
        return None

    for rec in db:
        if rec.get('id') == uid:
            return rec
    return None

