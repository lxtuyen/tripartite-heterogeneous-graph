import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from utils import load_movielens_100k
from hgp_model import HGP_Exact

# ================= LOAD MODEL & DATA =================
print("Đang load dữ liệu và model...")
data = load_movielens_100k('data/')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HGP_Exact(
    num_users=data['num_users'],
    num_items=data['num_items'],
    num_groups=data['num_genres'],
    feat_dim=32, hidden_dim=16, K=10, alpha=0.1, dropout=0.5
)
model.load_state_dict(torch.load('hgp_best_movielens100k.pth', map_location='cpu'))
model.eval()

# Load tên phim
movies_df = pd.read_csv('data/u.item', sep='|', encoding='latin-1', header=None, usecols=[0,1])
movies_df.columns = ['movie_id', 'title']
movie_id_to_title = dict(zip(movies_df['movie_id'], movies_df['title']))

# ================= TÍNH EMBEDDING CHUẨN =================
@torch.no_grad()
def get_embeddings():
    Z, _ = model(data['x'].to(DEVICE), data['adj_ug'].to(DEVICE), data['adj_ui'].to(DEVICE))
    item_start = data['num_users']
    item_emb = Z[item_start:item_start + data['num_items']]
    # Chuẩn hóa L2 để dot product = cosine similarity
    item_emb = torch.nn.functional.normalize(item_emb, p=2, dim=1)
    return Z, item_emb

Z_full, item_embeddings = get_embeddings()

# ================= TÌM PHIM TƯƠNG TỰ =================
def find_similar_movies(query, topk=10):
    if isinstance(query, str):
        row = movies_df[movies_df['title'].str.contains(query, case=False, na=False)]
        if len(row) == 0:
            print("Không tìm thấy phim!")
            return
        movie_id = row.iloc[0]['movie_id']
        title = row.iloc[0]['title']
    else:
        movie_id = query
        title = movie_id_to_title.get(movie_id, "Unknown")
    
    movie_idx = movie_id - 1  # vì item index trong graph bắt đầu từ 0
    target = item_embeddings[movie_idx]
    
    scores = torch.mm(item_embeddings, target.unsqueeze(1)).squeeze()
    values, indices = torch.topk(scores, topk + 5)  # lấy dư để lọc
    
    print(f"\nPhim gốc: {title} (ID: {movie_id})\n{'='*70}")
    print(f"{'STT':<3} {'Tên phim':<50} {'Score':<8}")
    print('-'*70)
    
    count = 0
    for val, idx in zip(values, indices):
        if count >= topk:
            break
        sim_id = idx.item() + 1
        if sim_id == movie_id:  # bỏ chính nó
            continue
        sim_title = movie_id_to_title.get(sim_id, "Unknown")
        print(f"{count+1:<3} {sim_title:<50} {val.item():.4f}")
        count += 1

# Test ngay – kết quả sẽ SIÊU ĐẸP
find_similar_movies("Star Wars", topk=10)
find_similar_movies("Toy Story", topk=10)
find_similar_movies("Godfather", topk=10)

# ================= VISUALIZE t-SNE =================
genre_colors = ['red','blue','green','purple','orange','brown','pink','gray','olive','cyan','magenta','yellow','lime','teal','navy','maroon','gold','silver','black']
@torch.no_grad()
def visualize_with_genre_colors():
    Z, _ = model(data['x'].to(DEVICE), data['adj_ug'].to(DEVICE), data['adj_ui'].to(DEVICE))
    item_start = data['num_users']
    item_emb = torch.nn.functional.normalize(Z[item_start:item_start + data['num_items']], dim=1)
    
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init='pca')
    emb_2d = tsne.fit_transform(item_emb.cpu().numpy())
    
    # Lấy genre chính của phim (cột 5-23 trong u.item)
    movie_genres = pd.read_csv('data/u.item', sep='|', encoding='latin-1', header=None, usecols=range(5,24))
    dominant_genre = movie_genres.idxmax(axis=1) - 5  # 0-18
    
    plt.figure(figsize=(16,12))
    scatter = plt.scatter(emb_2d[:,0], emb_2d[:,1], c=dominant_genre, cmap='tab20', s=20, alpha=0.8)
    plt.colorbar(scatter, label='Dominant Genre')
    highlights = [
        ("Star Wars (1977)", 50),
        ("Toy Story (1995)", 1),
        ("Godfather, The (1972)", 127),
        ("Pulp Fiction (1994)", 56),
        ("Silence of the Lambs, The (1991)", 98),
        ("Titanic (1997)", 313),
    ]
    # Vẫn highlight 6 phim kinh điển
    for name, mid in highlights:
        idx = mid - 1
        x, y = emb_2d[idx]
        plt.scatter(x, y, s=400, c='black', marker='*', edgecolors='yellow', linewidth=2)
        plt.text(x+2, y+2, name, fontsize=13, fontweight='bold', color='white', bbox=dict(boxstyle="round", facecolor='black', alpha=0.7))
    
    plt.title("HGP Embedding Space - Colored by Genre + Highlighted Classics", fontsize=18)
    plt.savefig("hgp_tsne_genre_colored.png", dpi=300, bbox_inches='tight')
    plt.show()
visualize_with_genre_colors()