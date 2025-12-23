import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

def load_movielens_100k(data_path='data/'):
    """
    Load MovieLens-100K thành heterogeneous graph: User - Item - Genre
    Chỉ dùng ratings >= 4 làm positive interaction (như CTR)
    """
    # 1. Load dữ liệu gốc
    ratings = pd.read_csv(
        data_path + 'u.data', sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    movies = pd.read_csv(
        data_path + 'u.item', sep='|', encoding='latin-1', header=None,
        names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + 
              [f'g{i}' for i in range(19)]
    )
    users = pd.read_csv(
        data_path + 'u.user', sep='|', 
        names=['user_id', 'age', 'gender', 'occupation', 'zip']
    )

    # 2. Chỉ giữ lại các genre thực sự tồn tại
    genre_names = [
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    genre_cols = [f'g{i}' for i in range(19)]

    # Tạo danh sách genre cho mỗi movie (list of strings)
    movie_genres = []
    for _, row in movies.iterrows():
        active = [genre_names[i] for i, col in enumerate(genre_cols) if row[col] == 1]
        movie_genres.append(active if active else ['unknown'])
    movies = movies[['movie_id']].copy()
    movies['genres'] = movie_genres

    # 3. Encode IDs
    user_le = LabelEncoder()
    item_le = LabelEncoder()
    ratings['user_idx'] = user_le.fit_transform(ratings['user_id'])
    ratings['item_idx'] = item_le.fit_transform(ratings['movie_id'])

    num_users = len(user_le.classes_)      # 943
    num_items = len(item_le.classes_)      # 1682

    # 4. Lấy tất cả genre duy nhất (19 cái)
    all_genres = sorted(set(g for genres in movie_genres for g in genres))
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}
    num_genres = len(all_genres)           # 19

    total_nodes = num_users + num_items + num_genres

    # 5. Tạo edges (chỉ dùng rating >= 4)
    pos_ratings = ratings[ratings['rating'] >= 4]

    # User - Item edges
    ui_edges = list(zip(
        pos_ratings['user_idx'],
        pos_ratings['item_idx'] + num_users
    ))

    # User - Genre edges (qua phim đã rate >= 4)
    ug_edges = []
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(item_le.classes_)}

    for _, row in pos_ratings.iterrows():
        movie_id = row['movie_id']
        item_idx = movie_id_to_idx[movie_id] + num_users
        genres = movies[movies['movie_id'] == movie_id]['genres'].iloc[0]
        for g in genres:
            genre_idx = genre_to_idx[g] + num_users + num_items
            ug_edges.append((row['user_idx'], genre_idx))

    # 6. Node features (dim = 32 là đủ cho 100k)
    feature_dim = 32

    # User features: occupation (21) + gender (2) + age_norm (1) → ~24 → pad to 32
    occ_dummy = pd.get_dummies(users['occupation'], prefix='occ')
    gender_dummy = pd.get_dummies(users['gender'], prefix='gen')
    age_norm = users['age'].values.reshape(-1, 1) / 100.0
    user_feat = np.hstack([occ_dummy.values, gender_dummy.values, age_norm])
    user_feat = np.hstack([user_feat, np.zeros((num_users, feature_dim - user_feat.shape[1]))])
    user_feat = torch.tensor(user_feat, dtype=torch.float)

    # Item features: multi-hot genre (19 dims) → pad to 32
    item_feat = np.zeros((num_items, num_genres))
    for idx, movie_id in enumerate(item_le.classes_):
        genres = movies[movies['movie_id'] == movie_id]['genres'].iloc[0]
        for g in genres:
            item_feat[idx, genre_to_idx[g]] = 1
    item_feat = np.hstack([item_feat, np.zeros((num_items, feature_dim - num_genres))])
    item_feat = torch.tensor(item_feat, dtype=torch.float)

    # Genre features: one-hot 19 → pad to 32
    genre_feat = np.eye(num_genres)
    genre_feat = np.hstack([genre_feat, np.zeros((num_genres, feature_dim - num_genres))])
    genre_feat = torch.tensor(genre_feat, dtype=torch.float)

    # Ghép lại
    x = torch.cat([user_feat, item_feat, genre_feat], dim=0)  # [total_nodes, 32]

    # 7. Tạo normalized adjacency (sparse, rất nhanh)
    def build_normalized_adj(edges, size):
        edges = torch.tensor(edges, dtype=torch.long).t()
        row, col = edges[0], edges[1]
        deg = torch.bincount(row, minlength=size).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(edges, norm, (size, size)).coalesce()

    adj_ui = build_normalized_adj(ui_edges, total_nodes)
    adj_ug = build_normalized_adj(ug_edges, total_nodes)

    # 8. Positive pairs để train (user_idx, item_idx) - dùng negative sampling trong train loop
    train_pos_pairs = [(u, i - num_users) for u, i in ui_edges]

    return {
        'x': x,                     # [total_nodes, 32]
        'adj_ui': adj_ui,           # User-Item normalized adj
        'adj_ug': adj_ug,           # User-Genre normalized adj
        'train_pos_pairs': train_pos_pairs,
        'num_users': num_users,     # 943
        'num_items': num_items,     # 1682
        'num_genres': num_genres,   # 19
        'total_nodes': total_nodes, # 943 + 1682 + 19 = 2644
    }

data = load_movielens_100k('data/')
print(data['x'].shape)
print(len(data['train_pos_pairs']))