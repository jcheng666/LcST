import torch
import torch.nn as nn

from model.llm import BaseModel, GPT2


class Transformer(BaseModel):
    def __init__(
        self,
        causal,
        lora,
        ln_grad,
        layers=None,
        emb_dim: int = 128,
        n_head: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        del lora, ln_grad

        depth = 3 if layers is None else layers
        self.emb_dim = emb_dim
        self.causal = bool(causal)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_head,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.out_norm = nn.LayerNorm(emb_dim)

    def _build_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.FloatTensor, attention_mask=None):
        del attention_mask
        mask = self._build_causal_mask(x.shape[1], x.device) if self.causal else None
        h = self.encoder(x, mask=mask)
        return self.out_norm(h)

    def getembedding(self, x: torch.FloatTensor):
        raise NotImplementedError("Transformer has no token embedding table")

    def gettokenizer(self):
        raise NotImplementedError("Transformer has no tokenizer")


def build_backbone(args):
    model_name = getattr(args, "model", "gpt2").lower()
    if model_name == "gpt2":
        return GPT2(args.causal, args.lora, args.ln_grad, args.llm_layers)
    if model_name in {"tiny_transformer", "transformer_small", "transformer"}:
        return Transformer(
            causal=args.causal,
            lora=args.lora,
            ln_grad=args.ln_grad,
            layers=args.llm_layers,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown model: {model_name}")
