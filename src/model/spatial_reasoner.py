import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from model.node_pe import NodePE


class DecodingLayer(nn.Module):
    def __init__(self, emb_dim, output_dim):
        super().__init__()

        hidden_size = (emb_dim + output_dim) * 2 // 3
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, llm_hidden: Float[Tensor, "C D"]) -> Float[Tensor, "C H"]:
        return self.fc(llm_hidden)


class SpatialReasoner(nn.Module):
    @beartype
    def __init__(
        self,
        basemodel: nn.Module,
        sample_len: int,
        output_len: int,
        output_dim: int,
        n_soft_prompt: int = 8,
        node_pe_k: int = 16,
        node_pe_enabled: bool = True,
    ):
        super().__init__()
        self.basemodel = basemodel
        self.emb_dim = basemodel.emb_dim
        self.output_len = output_len
        self.output_dim = output_dim
        self.node_pe_enabled = node_pe_enabled and node_pe_k > 0

        self.out_mlp = DecodingLayer(
            emb_dim=self.emb_dim,
            output_dim=output_dim * output_len,
        )
        self.layer_norm = nn.LayerNorm(self.emb_dim)

        # 由 ``init_sep_from_eos()`` 在父模块上设备后填充
        self.sep_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
        self.soft_prompt = nn.Parameter(torch.randn(1, n_soft_prompt, self.emb_dim))

        if self.node_pe_enabled:
            self.node_pe = NodePE(k=node_pe_k, emb_dim=self.emb_dim)

    @beartype
    def init_sep_from_eos(self) -> bool:
        """从 backbone 的 EOS token embedding 初始化 sep_token。"""
        try:
            tokenizer = getattr(self.basemodel, "gettokenizer")()
            eos_id = tokenizer.eos_token_id
        except (AttributeError, NotImplementedError):
            return False

        if eos_id is None:
            return False

        device = self.sep_token.device
        eos_token = torch.tensor([[eos_id]], device=device, dtype=torch.long)
        with torch.no_grad():
            eos_emb = getattr(self.basemodel, "getembedding")(eos_token).to(device=device, dtype=self.sep_token.dtype)
            self.sep_token.copy_(eos_emb.view_as(self.sep_token))
        return True

    @torch.no_grad()
    def encode_graph_pe(self, adj_mx, device: torch.device) -> Tensor | None:
        """Compute projected Laplacian PE for all nodes of a graph."""
        if not self.node_pe_enabled:
            return None
        return self.node_pe.encode_graph(adj_mx, device)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        target_tok: Float[Tensor, "C 1 D"],
        aux_tok: Float[Tensor, "C K D"],
        target_ids: Int[Tensor, "C"] | None = None,
        aux_ids: Int[Tensor, "C K"] | None = None,
        node_pe: Float[Tensor, "N D"] | None = None,
    ) -> Float[Tensor, "C H"]:
        """Run a batch of (target, aux) token tuples through the backbone."""
        B = target_tok.shape[0]

        if node_pe is not None and target_ids is not None:
            target_tok = target_tok + node_pe[target_ids].unsqueeze(1)
            if aux_ids is not None:
                aux_tok = aux_tok + node_pe[aux_ids]

        sep = self.sep_token.expand(B, -1, -1)
        soft_prompt = self.soft_prompt.expand(B, -1, -1)
        hidden = torch.cat([soft_prompt, aux_tok, sep, target_tok], dim=1)

        s_state = self.basemodel(hidden)
        s_target = s_state[:, -1, :]
        s_target = self.layer_norm(s_target + target_tok.squeeze(1))
        return self.out_mlp(s_target)
