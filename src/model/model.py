from typing import Optional, Tuple

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from model.graph_context import GraphContext
from model.spatial_reasoner import SpatialReasoner
from model.temporal_tokenizer import TemporalTokenizer
from model.window_encoder import EncodedBank, WindowEncoder


class STALLM(nn.Module):
    @beartype
    def __init__(
        self,
        basemodel: nn.Module,
        sample_len: int,
        output_len: int,
        input_dim: int,
        output_dim: int,
        adj_mx,
        dropout: float = 0,
        unit_len: int = 36,
        token_dim: int = 96,
        token_head: int = 2,
        n_aux: int = 16,
        aux_neighbor_order: str = "topological",
        aux_neighbor_fill: str = "higher_order",
        backbone_capacity: int = 4096,
        node_pe_k: int = 16,
        node_pe_enabled: bool = True,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.sample_len = sample_len
        self.output_len = output_len
        self.n_aux = n_aux
        self.backbone_capacity = backbone_capacity
        self.emb_dim = basemodel.emb_dim

        self._aux_neighbor_order = aux_neighbor_order
        self._aux_neighbor_fill = aux_neighbor_fill

        self.tokenizer = TemporalTokenizer(
            sample_len=sample_len,
            input_dim=input_dim,
            unit_len=unit_len,
            token_dim=token_dim,
            token_head=token_head,
            emb_dim=self.emb_dim,
            dropout=dropout,
        )
        self.reasoner = SpatialReasoner(
            basemodel=basemodel,
            sample_len=sample_len,
            output_len=output_len,
            output_dim=output_dim,
            node_pe_k=node_pe_k,
            node_pe_enabled=node_pe_enabled,
        )

        self._graph_ctxs: dict[str, GraphContext] = {}
        self._current_ctx: Optional[GraphContext] = None
        self._init_graph_ctx("default", adj_mx)
        self.window_encoder = WindowEncoder(self.tokenizer, self._current_graph_ctx())

    def _init_graph_ctx(self, graph_key: str, adj_mx) -> None:
        ctx = GraphContext(
            adj_mx=adj_mx,
            n_aux=self.n_aux,
            neighbor_order=self._aux_neighbor_order,
            neighbor_fill=self._aux_neighbor_fill,
            pe_k=self.reasoner.node_pe.k if hasattr(self.reasoner, 'node_pe') else 0,
            node_pe_enabled=getattr(self.reasoner, 'node_pe_enabled', False),
        )
        self._graph_ctxs[graph_key] = ctx
        self._current_ctx = ctx

    @beartype
    def add_graph(self, graph_key: str, adj_mx) -> None:
        self._init_graph_ctx(graph_key, adj_mx)

    @beartype
    def set_graph(self, graph_key: str, adj_mx=None) -> None:
        """Activate a graph (GraphContext manages PE lazily)."""
        if graph_key not in self._graph_ctxs:
            if adj_mx is None:
                raise KeyError(f"unknown graph '{graph_key}' and no adj_mx provided")
            self._init_graph_ctx(graph_key, adj_mx)
        self._current_ctx = self._graph_ctxs[graph_key]

    def _current_graph_ctx(self) -> GraphContext:
        assert self._current_ctx is not None, "no graph is active; call set_graph(...) first"
        return self._current_ctx

    @beartype
    def resample_aux_pools(self) -> None:
        self._current_graph_ctx().resample()

    @beartype
    def sample_aux_pool_sets(
        self,
        n_sets: int,
        seed: Optional[int] = None,
    ):
        return self._current_graph_ctx().sample_sets(n_sets, seed=seed)

    def set_aux_pools(self, aux_pools) -> None:
        self._current_graph_ctx().set_pools(aux_pools)

    @beartype
    def sample_node_aux(self, node_ids: Int[Tensor, "C"], device: torch.device) -> Int[Tensor, "C K"]:
        """Get aux node IDs for the given target nodes via GraphContext."""
        return self._current_graph_ctx().sample_neighbors(node_ids, device)

    @beartype
    def init_sep_from_eos(self) -> bool:
        return self.reasoner.init_sep_from_eos()

    def _node_pe(self, device: torch.device) -> Optional[Tensor]:
        """Get projected Laplacian PE for the current graph.

        GraphContext computes raw eigenvectors (CPU); SpatialReasoner's NodePE
        projects them to embedding space on the correct device.
        """
        ctx = self._current_graph_ctx()
        raw = ctx.raw_eigvecs()
        if raw is None:
            return None
        eigvecs = raw.to(device=device)
        return self.reasoner.node_pe(eigvecs)

    def _reasoner_dispatch(self, target_tok, aux_tok, target_ids, aux_ids, node_pe):
        """Forward through the reasoner, chunking when the batch exceeds capacity."""
        C = target_tok.shape[0]
        if C <= self.backbone_capacity:
            return self.reasoner(
                target_tok=target_tok, aux_tok=aux_tok,
                target_ids=target_ids, aux_ids=aux_ids, node_pe=node_pe,
            )
        outputs = []
        for start in range(0, C, self.backbone_capacity):
            end = start + self.backbone_capacity
            outputs.append(self.reasoner(
                target_tok=target_tok[start:end], aux_tok=aux_tok[start:end],
                target_ids=target_ids[start:end], aux_ids=aux_ids[start:end],
                node_pe=node_pe,
            ))
        return torch.cat(outputs, dim=0)

    @jaxtyped(typechecker=beartype)
    def encode(
        self,
        x: Float[Tensor, "B N TF"],
        target_node_ids: Int[Tensor, "C"] | None = None,
    ) -> EncodedBank:
        """Encode windows through WindowEncoder (sparse or full)."""
        return self.window_encoder.encode(x, target_node_ids)

    def predict(
        self,
        bank: EncodedBank,
        sample_ids: Int[Tensor, "C"],
        node_ids: Int[Tensor, "C"],
    ) -> Float[Tensor, "C H"]:
        """Predict from an EncodedBank for given (sample, node) pairs."""
        sample_ids = sample_ids.to(device=bank.tokens.device, dtype=torch.long)
        node_ids = node_ids.to(device=bank.tokens.device, dtype=torch.long)
        aux_ids = self.sample_node_aux(node_ids, bank.tokens.device)
        target_tok, aux_tok = bank.query(sample_ids, node_ids, aux_ids)
        node_pe = self._node_pe(bank.tokens.device)
        return self._reasoner_dispatch(target_tok, aux_tok, node_ids, aux_ids, node_pe)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "B N TF"],
        target_ids: Optional[Int[Tensor, "C"]] = None,
    ) -> Float[Tensor, "B C H"]:
        B, N, _ = x.shape
        if target_ids is None:
            target_ids = torch.arange(N, device=x.device)
        C = target_ids.numel()
        bank = self.encode(x, target_ids)
        sample_ids = torch.arange(B, device=x.device).unsqueeze(1).expand(B, C).reshape(-1)
        node_ids = target_ids.unsqueeze(0).expand(B, C).reshape(-1)
        return self.predict(bank, sample_ids, node_ids).view(B, C, -1)

    def grad_state_dict(self):
        params_to_save = filter(lambda p: p[1].requires_grad, self.named_parameters())
        save_list = [p[0] for p in params_to_save]
        return {name: param for name, param in self.state_dict().items() if name in save_list}

    @beartype
    def save(self, path: str) -> None:
        torch.save(self.grad_state_dict(), path)

    @beartype
    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path), strict=False)

    def params_num(self) -> Tuple[int, int]:
        total_params = sum(p.numel() for p in self.parameters())
        total_params += sum(p.numel() for p in self.buffers())
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, total_trainable_params
