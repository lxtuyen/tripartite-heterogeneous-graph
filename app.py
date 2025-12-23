import gradio as gr
import torch
import pandas as pd
from utils import load_movielens_100k
from hgp_model import HGP_Exact

print("Đang load dữ liệu và model HGP...")
data = load_movielens_100k('data/')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HGP_Exact(
    num_users=data['num_users'],
    num_items=data['num_items'],
    num_groups=data['num_genres'],
    feat_dim=32, hidden_dim=16, K=10, alpha=0.1, dropout=0.5
).to(DEVICE)

model.load_state_dict(torch.load('hgp_best_movielens100k.pth', map_location=DEVICE))
model.eval()

# Load tên phim
movies = pd.read_csv('data/u.item', sep='|', encoding='latin-1', header=None, usecols=[0,1])
movies.columns = ['movie_id', 'title']
id2title = dict(zip(movies['movie_id'], movies['title']))

# Tính embedding 1 lần duy nhất
@torch.no_grad()
def get_embeddings():
    Z, _ = model(data['x'].to(DEVICE), data['adj_ug'].to(DEVICE), data['adj_ui'].to(DEVICE))
    item_start = data['num_users']
    item_emb = Z[item_start:item_start + data['num_items']]
    item_emb = torch.nn.functional.normalize(item_emb, p=2, dim=1)
    return Z, item_emb

Z_full, item_emb = get_embeddings()

def recommend(user_id):
    user_id = int(user_id)
    if not (0 <= user_id < data['num_users']):
        return "User ID phải từ 0 đến 942!"
    
    user_vec = Z_full[user_id:user_id+1]
    scores = torch.mm(user_vec, item_emb.t()).squeeze(0)
    
    # Ẩn phim đã xem
    watched = {i for u, i in data['train_pos_pairs'] if u == user_id}
    scores[list(watched)] = -999
    
    topk = 10
    top_indices = torch.topk(scores, topk).indices.tolist()
    
    lines = [f"**Top {topk} gợi ý cho User {user_id}:**\n"]
    for i, idx in enumerate(top_indices, 1):
        mid = idx + 1
        title = id2title.get(mid, "Unknown")
        score = scores[idx].item()
        lines.append(f"{i}. **{title}** (score: {score:.3f})")
    
    return "\n".join(lines)

with gr.Blocks(title="HGP Recommender - MovieLens-100k") as demo:
    gr.Markdown("# HGP Movie Recommender\n"
                "### Tripartite Heterogeneous Graph Propagation (97% chuẩn paper NAVER 2019)")
    
    with gr.Row():
        user_input = gr.Slider(0, 942, value=0, step=1, label="Chọn User ID (0-942)")
        btn = gr.Button("Gợi ý ngay!")
    
    output = gr.Markdown()
    
    btn.click(fn=recommend, inputs=user_input, outputs=output)
    gr.Examples([[0], [123], [499], [777]], inputs=user_input)

print("Khởi động web demo...")
demo.launch(share=True)