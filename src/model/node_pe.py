import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


def laplacian_eigvecs(
    adj_mx,
    k: int,
) -> Tensor:
    """Compute top-k eigenvectors of the normalised graph Laplacian."""
    if isinstance(adj_mx, Tensor):
        adj_mx = adj_mx.detach().cpu().numpy()

    A = np.asarray(adj_mx, dtype=np.float64)
    # Symmetrise (undirected graph)
    A = np.maximum(A, A.T)
    N = A.shape[0]

    # Normalised Laplacian  L_norm = I - D^{-1/2} A D^{-1/2}
    # Eigenvalues constrained to [0, 2] — better zero-shot transfer across graphs.
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    L = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt

    if N <= 1:
        eigvecs = np.zeros((N, k), dtype=np.float32)
        return torch.from_numpy(eigvecs)

    # eigh returns eigenvalues in ascending order.  λ₀ = 0 with constant eigenvector
    # — skip it, as it carries no structural information.
    effective_k = min(k, N - 2)  # -1 for max eigenvectors, -1 to skip constant
    if effective_k <= 0:
        eigvecs = np.zeros((N, k), dtype=np.float32)
        return torch.from_numpy(eigvecs)

    eigvals, eigvecs = np.linalg.eigh(L)
    eigvecs = eigvecs[:, 1 : 1 + effective_k].astype(np.float64)  # skip λ₀

    for i in range(effective_k):
        col = eigvecs[:, i]
        max_idx = int(np.argmax(np.abs(col)))  # type: ignore[arg-type]
        if col[max_idx] < 0:
            col *= -1.0

    # Pad to k if graph is small
    if effective_k < k:
        pad = np.zeros((N, k - effective_k), dtype=np.float64)
        eigvecs = np.concatenate([eigvecs, pad], axis=1)

    return torch.from_numpy(eigvecs.astype(np.float32))


class NodePE(nn.Module):
    """Project Laplacian eigenvectors into the token embedding space."""

    def __init__(self, k: int = 16, emb_dim: int = 768):
        super().__init__()
        self.k = k
        self.emb_dim = emb_dim
        self.proj = nn.Sequential(
            nn.Linear(k, emb_dim // 2),
            nn.GELU(),
            nn.Linear(emb_dim // 2, emb_dim),
        )

    def forward(self, eigvec: Tensor) -> Tensor:
        return self.proj(eigvec)

    @torch.no_grad()
    def encode_graph(self, adj_mx, device: torch.device) -> Tensor:
        """Convenience: compute eigvecs + project for one graph."""
        eigvecs = laplacian_eigvecs(adj_mx, self.k).to(device=device)
        return self.forward(eigvecs)
