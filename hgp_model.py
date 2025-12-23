# hgp_exact_paper.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HGP_Exact(nn.Module):
    """
    Heterogeneous Graph Propagation - EXACTLY as in paper 1908.02569
    Tested on MovieLens-100k with Group=Genre
    """
    def __init__(self, num_users, num_items, num_groups, feat_dim=16, hidden_dim=16, K=10, alpha=0.1, dropout=0.5):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_groups = num_groups
        self.hidden_dim = hidden_dim
        self.K = K
        self.alpha = alpha

        total_nodes = num_users + num_items + num_groups

        # === 4.1: Predicting networks riêng cho từng loại node ===
        self.f_user  = nn.Linear(feat_dim, hidden_dim)
        self.f_item  = nn.Linear(feat_dim, hidden_dim)
        self.f_group = nn.Linear(feat_dim, hidden_dim)

        # === Propagation weight W_H^(k) - trong paper dùng chung cho mọi layer ===
        self.W_h = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W_h)

        # === Attention như Transformer (Eq. 3-4) ===
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim // 2)   # paper: dim WV = (m,8)
        self.final_merge = nn.Linear(hidden_dim, hidden_dim)  # concat 2 × (m//2) → m

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.f_user, self.f_item, self.f_group]:
            nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.final_merge.weight)

    def forward(self, x, adj_gu, adj_ui):
        """
        x:       [total_nodes, 16]
        adj_gu:  normalized Group-User adj (sparse)
        adj_ui:  normalized User-Item adj (sparse)
        """
        device = x.device
        N = x.size(0)

        # === Step 1: H = f_theta(X) với ReLU ===
        xu = F.relu(self.f_user(x[:self.num_users]))
        xi = F.relu(self.f_item(x[self.num_users:self.num_users + self.num_items]))
        xg = F.relu(self.f_group(x[-self.num_groups:]))

        H = torch.zeros(N, self.hidden_dim, device=device)
        H[:self.num_users] = xu
        H[self.num_users:self.num_users + self.num_items] = xi
        H[-self.num_groups:] = xg

        # === Step 2: 2 propagation riêng biệt (Eq.2) ===
        def propagate(adj, Z_init):
            Z = Z_init
            for k in range(self.K):
                Z_new = (1 - self.alpha) * F.relu(torch.sparse.mm(adj, Z) @ self.W_h) + self.alpha * Z_init
                Z = self.dropout(Z_new)
            return Z

        Z_gu = propagate(adj_gu, H)   # từ Group-User graph
        Z_ui = propagate(adj_ui, H)   # từ User-Item graph

        # === Step 3: Attention kết hợp 2 embeddings (Eq.3-4) ===
        # Stack 2 embeddings: [N, 2, hidden_dim]
        Y = torch.stack([Z_gu, Z_ui], dim=1)          # shape (N, 2, m)

        Q = self.W_q(Y)                               # (N,2,m)
        K = self.W_k(Y)
        V = self.W_v(Y)                               # (N,2,m//2)

        attn_scores = torch.softmax(torch.bmm(Q, K.transpose(1,2)) / (self.hidden_dim**0.5), dim=-1)
        Y_att = torch.bmm(attn_scores, V)             # (N,2,m//2)
        Y_att = Y_att.reshape(N, -1)                  # (N, m)
        Z_final = self.final_merge(Y_att)             # (N, m)

        return Z_final, H  # Z_final: final embedding, H: raw feature projection (dùng cho Eq.5)

    def predict_ctr(self, Z, X_raw, user_idx, item_idx):
        """
        Eq.5: p_ij = sigmoid( z_u^T z_i + x_u^T x_i )
        """
        zu = Z[user_idx]
        zi = Z[self.num_users + item_idx]
        xu = X_raw[user_idx]
        xi = X_raw[self.num_users + item_idx]

        dot_z = (zu * zi).sum(dim=-1)
        dot_x = (xu * xi).sum(dim=-1)
        return torch.sigmoid(dot_z + dot_x)