import torch
import torch.nn as nn
from pathlib import Path


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, attention_mask=None):
        raise NotImplementedError("error")

    def getembedding(self, x):
        raise NotImplementedError("error")

    def gettokenizer(self):
        raise NotImplementedError("error")


class GPT2(BaseModel):
    def __init__(self, causal, lora, ln_grad, layers=None):
        super().__init__()

        causal = bool(causal)

        self.emb_dim = 768

        self.llm, self.tokenizer = self._load_model()

        if layers is not None:
            self.transformer.h = self.transformer.h[:layers]

        self.causal = causal

        for name, param in self.llm.named_parameters():
            param.requires_grad_(False)

        if lora:
            self._install_lora()

        if ln_grad:
            for name, param in self.llm.named_parameters():
                if "ln" in name or "wpe" in name:
                    param.requires_grad = True


    @property
    def transformer(self):
        if hasattr(self.llm, "base_model") and hasattr(self.llm.base_model, "model"):
            base_model = self.llm.base_model.model
            return getattr(base_model, "transformer", base_model)
        return getattr(self.llm, "transformer", self.llm)

    def _install_lora(self):
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=16,
            target_modules=["q_attn", "c_attn"],
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
        )
        self.llm = get_peft_model(self.llm, lora_config)

    def _load_model(self):
        try:
            from modelscope import AutoTokenizer
            from modelscope.models import Model

            model = Model.from_pretrained("AI-ModelScope/gpt2", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained("AI-ModelScope/gpt2", trust_remote_code=True)
            return model, tokenizer
        except Exception:
            from transformers import AutoModel, AutoTokenizer

            cache_path = Path.home() / ".cache/modelscope/hub/models/AI-ModelScope/gpt2"
            model_name = str(cache_path) if cache_path.exists() else "gpt2"
            model = AutoModel.from_pretrained(model_name, local_files_only=cache_path.exists())
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=cache_path.exists())
            return model, tokenizer

    def forward(self, x: torch.FloatTensor, attention_mask=None):
        out = self.transformer(
            inputs_embeds=x,
            attention_mask=attention_mask,
        ).last_hidden_state
        return out

    def getembedding(self, x: torch.FloatTensor):
        return self.transformer.wte(x)

    def gettokenizer(self):
        return self.tokenizer

