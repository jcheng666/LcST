"""WindowEncoder — unified sparse/full window encoding with EncodedBank.

Replaces the dual encode_window / encode_window_sparse + predict_from_sparse_bank /
predict_target_instances code paths with a single encode() → EncodedBank → predict() flow.
"""

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from model.graph_context import GraphContext


class EncodedBank:
    """Holds a (possibly sparse) token bank and an index map for lookups.

    ``tokens`` has shape (B, M, D) where M is the number of *encoded* nodes.
    ``index_map`` has shape (N,) mapping global node_id → column in tokens
    (-1 means "not in bank").
    """

    def __init__(self, tokens: Tensor, index_map: Tensor):
        self.tokens = tokens
        self.index_map = index_map

    @jaxtyped(typechecker=beartype)
    def query(
        self,
        sample_ids: Int[Tensor, "C"],
        node_ids: Int[Tensor, "C"],
        aux_ids: Int[Tensor, "C K"],
    ):
        """Return (target_tok, aux_tok) ready for reasoner dispatch.

        Remaps sparse node indices through self.index_map.
        """
        col_target = self.index_map[node_ids]
        col_aux = self.index_map[aux_ids]

        target_tok = self.tokens[sample_ids, col_target].reshape(node_ids.numel(), 1, -1)
        aux_tok = self.tokens[sample_ids.unsqueeze(1), col_aux]
        return target_tok, aux_tok


class WindowEncoder:
    """Encodes sliding-window inputs through TemporalTokenizer, using GraphContext
    for auto-sparse auxiliary-node selection."""

    @beartype
    def __init__(self, tokenizer, graph_ctx: GraphContext):
        self.tokenizer = tokenizer
        self.graph_ctx = graph_ctx

    @jaxtyped(typechecker=beartype)
    def encode(
        self,
        windows: Float[Tensor, "B N TF"],
        target_node_ids: Int[Tensor, "C"] | None = None,
        mask: Float[Tensor, "B N TF"] | None = None,
    ) -> EncodedBank:
        """Encode windows, optionally sparsely.

        - target_node_ids=None → full encode (all N nodes).
        - target_node_ids given → compute aux neighbors → encode only needed nodes.
        - mask: optional observation mask for tokenizer (same shape as windows).
        """
        B, N, _ = windows.shape
        device = windows.device

        if target_node_ids is None:
            # ---- full encode ----
            mask_flat = mask.view(B * N, -1) if mask is not None else None
            flat = self.tokenizer._forward_flat(windows.view(B * N, -1), mask_flat)
            tokens = flat.view(B, N, -1)
            index_map = torch.arange(N, device=device)
            return EncodedBank(tokens, index_map)

        # ---- sparse encode ----
        target_node_ids = target_node_ids.to(device=device, dtype=torch.long)
        aux_ids = self.graph_ctx.sample_neighbors(target_node_ids, device)
        needed_node_ids = torch.unique(
            torch.cat([target_node_ids, aux_ids.reshape(-1)])
        )

        # Encode only needed nodes via tokenizer._forward_flat
        x_needed = windows[:, needed_node_ids, :]  # (B, M, TF)
        M = needed_node_ids.numel()
        mask_needed = None
        if mask is not None:
            mask_needed = mask[:, needed_node_ids, :]
        mask_flat = mask_needed.reshape(B * M, -1) if mask_needed is not None else None
        encoded = self.tokenizer._forward_flat(x_needed.reshape(B * M, -1), mask_flat)
        tokens = encoded.view(B, M, -1)

        # Build index_map: global node_id → column in tokens
        index_map = torch.full((N,), -1, dtype=torch.long, device=device)
        index_map[needed_node_ids] = torch.arange(M, device=device)

        return EncodedBank(tokens, index_map)