import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor


class TemporalTokenizer(nn.Module):
    def __init__(self, sample_len, input_dim, unit_len, token_dim, token_head, emb_dim, dropout):
        super().__init__()

        assert sample_len % unit_len == 0
        assert token_dim % token_head == 0

        self.unit_dim = unit_len * input_dim
        self.token_dim = token_dim

        self.ffn1 = nn.Sequential(
            nn.Linear(self.unit_dim * 2 - 1, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, token_dim),
        )

        self.head = token_head
        self.head_dim = token_dim // token_head
        self.mha1 = nn.MultiheadAttention(embed_dim=token_dim, num_heads=token_head, batch_first=True, dropout=dropout)
        self.pe = nn.Parameter(torch.randn(1, sample_len // unit_len, token_dim))
        self.attn_ln1 = nn.LayerNorm(token_dim)

        self.attn_ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.ffn_ln = nn.LayerNorm(token_dim)

        self.mha2 = nn.MultiheadAttention(embed_dim=token_dim, num_heads=token_head, batch_first=True, dropout=dropout)
        self.attn_ln2 = nn.LayerNorm(token_dim)

        self.out_ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.ReLU(),
            nn.Linear(token_dim * 4, emb_dim),
        )

        self.ln = nn.LayerNorm(emb_dim)
        seq_len = sample_len // unit_len
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    @jaxtyped(typechecker=beartype)
    def _forward_flat(self, x: Float[Tensor, "rows TF"]) -> Float[Tensor, "rows D"]:
        x = x.view(x.shape[0], -1, self.unit_dim)
        x_grad = x[..., 1:] - x[..., :-1]
        x = torch.concat((x, x_grad), dim=-1)
        h = self.ffn1(x)

        out, _ = self.mha1(
            query=h + self.pe,
            key=h + self.pe,
            value=h,
            attn_mask=self.causal_mask,
            need_weights=False,
        )
        h = self.attn_ln1(h + out)

        h = self.ffn_ln(h + self.attn_ffn(h))

        out, _ = self.mha2(query=h[:, -1:, :], key=h, value=h, need_weights=False)
        h = self.attn_ln2(h[:, -1:, :] + out)

        out = h.view(x.shape[0], -1)
        out = self.out_ffn(out)
        out = self.ln(out)
        return out

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B N TF"]) -> Float[Tensor, "B N D"]:
        B, N, _ = x.shape
        out = self._forward_flat(x.view(B * N, -1))
        return out.view(B, N, -1)

    @jaxtyped(typechecker=beartype)
    def forward_sparse(
        self,
        x: Float[Tensor, "B N TF"],
        flat_indices: Int[Tensor, "M"],
    ) -> Float[Tensor, "M D"]:
        selected = x.reshape(-1, x.shape[-1]).index_select(0, flat_indices)
        return self._forward_flat(selected)

    def encode_sparse(self, x, needed_node_ids):
        """仅编码指定节点在批次中所有样本的 token。"""
        x_needed = x[:, needed_node_ids, :]  # (B, M, TF)
        B, M, _ = x_needed.shape
        out = self._forward_flat(x_needed.reshape(B * M, -1))
        return out.view(B, M, -1)
