# main_exact.py
import numpy as np
import torch
from torch.optim import Adam
from utils import load_movielens_100k
from hgp_model import HGP_Exact
from tqdm import tqdm
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ================== LOAD DATA ==================
data = load_movielens_100k('data/')
x = data['x'].to(DEVICE)
adj_gu = data['adj_ug'].to(DEVICE)
adj_ui = data['adj_ui'].to(DEVICE)
train_pairs = data['train_pos_pairs']

print(f"Nodes: {data['total_nodes']} | Positives: {len(train_pairs)}")

# ================== MODEL CHUẨN 100% PAPER ==================
model = HGP_Exact(
    num_users=data['num_users'],
    num_items=data['num_items'],
    num_groups=data['num_genres'],
    feat_dim=32,       # bạn đang dùng 32-dim feature → để nguyên
    hidden_dim=16,     # đúng paper
    K=10,              # đúng paper
    alpha=0.1,         # đúng paper
    dropout=0.5
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# ================== HELPER ==================
pos_set = {(u, i) for u, i in train_pairs}

def sample_negatives(users, n=1):
    negs = []
    for u in users:
        for _ in range(n):
            j = random.randint(0, data['num_items'] - 1)
            while (u, j) in pos_set:
                j = random.randint(0, data['num_items'] - 1)
            negs.append(j)
    return torch.tensor(negs, device=DEVICE)

def bpr_loss(pos, neg):
    return -torch.log(torch.sigmoid(pos - neg) + 1e-12).mean()

# ================== EVALUATION (Recall@20, NDCG@20) ==================
@torch.no_grad()
def test_recall_ndcg(topk=20):
    model.eval()
    Z, _ = model(x, adj_gu, adj_ui)
    
    user_emb = Z[:data['num_users']]
    item_emb = Z[data['num_users']:data['num_users'] + data['num_items']]
    
    # Tính score cho tất cả user-item pairs
    scores = torch.mm(user_emb, item_emb.t())  # [num_users, num_items]
    
    _, indices = torch.topk(scores, topk, dim=1)  # [num_users, topk]
    
    hits = 0
    ndcg = 0.0
    total_pos = 0
    
    # Ground truth cho mỗi user
    user_pos_dict = {}
    for u, i in train_pairs:
        user_pos_dict.setdefault(u, set()).add(i)
    
    for u in range(data['num_users']):
        if u not in user_pos_dict:
            continue
        pos_items = user_pos_dict[u]
        ranked = indices[u].cpu().tolist()
        
        # Hit
        hit = len(set(ranked) & pos_items)
        hits += hit
        total_pos += len(pos_items)
        
        # NDCG
        for rank, item in enumerate(ranked, 1):
            if item in pos_items:
                ndcg += 1 / np.log2(rank + 1)
    
    recall = hits / total_pos if total_pos > 0 else 0
    ndcg_val = ndcg / total_pos if total_pos > 0 else 0
    return recall, ndcg_val

# ================== TRAINING + SAVE MODEL ==================
best_recall = 0.0
print("Start training HGP (exact paper implementation)...")

for epoch in range(1, 51):
    model.train()
    random.shuffle(train_pairs)
    total_loss = 0.0
    batch_size = 1024

    for i in range(0, len(train_pairs), batch_size):
        batch = train_pairs[i:i + batch_size]
        if len(batch) < 2: continue

        users = torch.tensor([p[0] for p in batch], device=DEVICE)
        pos_items = torch.tensor([p[1] for p in batch], device=DEVICE)
        neg_items = sample_negatives(users.tolist())

        optimizer.zero_grad()
        Z, H = model(x, adj_gu, adj_ui)

        pos_score = model.predict_ctr(Z, x, users, pos_items)
        neg_score = model.predict_ctr(Z, x, users, neg_items)

        loss = bpr_loss(pos_score, neg_score)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss * batch_size / len(train_pairs)

    # Đánh giá mỗi 5 epoch
    if epoch % 5 == 0 or epoch == 50:
        recall, ndcg = test_recall_ndcg(topk=20)
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | "
              f"Recall@20: {recall:.4f} | NDCG@20: {ndcg:.4f}")
        
        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), 'hgp_best_movielens100k.pth')
            print(f"    → New best model saved! (Recall@20 = {recall:.4f})")

print("\n" + "="*60)
print("TRAINING HOÀN TẤT!")
print(f"Best Recall@20: {best_recall:.4f}")
print("File model đã lưu: hgp_best_movielens100k.pth")
print("Bạn có thể load lại bằng:")
print("    model.load_state_dict(torch.load('hgp_best_movielens100k.pth'))")
print("="*60)