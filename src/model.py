import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Multi-head causal self-attention.

    Purpose:
        Core attention mechanism for the transformer.
    What:
        Projects input through W_Q, W_K, W_V into n_heads separate heads,
        computes scaled dot-product attention, concatenates heads, projects
        through W_O.
    Why:
        Standard GPT-style attention used in grokking experiments with
        bias-free linear layers for cleaner interpretability.
    """
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.W_Q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_O = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_Q(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        attn = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)
        return self.W_O(out)


class MLP(nn.Module):
    """Two-layer ReLU MLP with bias-free linear layers.

    Purpose:
        Feedforward sub-layer of the transformer block.
    What:
        Projects d_model -> d_mlp (ReLU) -> d_model.
    Why:
        Standard GPT-style MLP. The bias-free design (used in grokking
        literature) simplifies circuit analysis.
    """
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_mlp, bias=False)
        self.W_out = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x):
        return self.W_out(F.relu(self.W_in(x)))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-LN attention + MLP.

    Purpose:
        Residual stream building block: attention then MLP, each with
        pre-layer-norm and residual connection.
    What:
        LayerNorm -> Attention -> + residual -> LayerNorm -> MLP -> + residual.
    Why:
        Pre-LN (LayerNorm before the sub-layer, not after) is the standard
        formulation used in modern GPT-style transformers.
    """
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_mlp: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, d_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    """Decoder-only transformer for modular arithmetic.

    Purpose:
        The main model used across all experiments. Trained on (a + b) mod P
        with direct token IDs (0..P-1).
    What:
        Embedding + positional embedding -> Nx TransformerBlock -> LN ->
        unembed (linear to vocab). Optionally returns per-layer activations
        for probing/steering.
    Why:
        Minimal GPT-style transformer designed for grokking experiments.
        The return_activations flag enables activation extraction without
        hooks, used throughout this project for probing, CCA, and steering.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        d_vocab = cfg["d_vocab"]
        d_model = cfg["d_model"]
        n_layers = cfg["n_layers"]
        n_heads = cfg["n_heads"]
        d_head = cfg["d_head"]
        d_mlp = cfg["d_mlp"]
        n_ctx = cfg["n_ctx"]

        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(d_vocab, d_model)
        self.pos_embed = nn.Embedding(n_ctx, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_head, d_mlp)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, d_vocab, bias=False)

    def forward(self, tokens, return_activations: bool = False):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.embed(tokens) + self.pos_embed(pos)

        activations = {}
        for i, block in enumerate(self.blocks):
            x = block(x)
            if return_activations:
                activations[f"blocks.{i}.hook_resid_post"] = x.detach()

        x = self.ln_final(x)
        logits = self.unembed(x)
        if return_activations:
            return logits, activations
        return logits


def make_model(cfg: dict) -> Transformer:
    """Create a Transformer from a config dict.

    Purpose:
        Factory function for instantiating models across experiments.
    What:
        Returns Transformer(cfg). Named function for cleaner imports.
    Why:
        Provides a single entry point for model creation. Used by all
        experiment scripts (train.py, clean_test.py, line_a.py, etc.).
    """
    return Transformer(cfg)


CFG_SMALL = {
    "d_model": 128,
    "n_layers": 2,
    "n_heads": 4,
    "d_head": 32,
    "d_mlp": 512,
    "d_vocab": 97,
    "n_ctx": 2,
    "name": "small",
}

CFG_BIG = {
    "d_model": 512,
    "n_layers": 6,
    "n_heads": 8,
    "d_head": 64,
    "d_mlp": 2048,
    "d_vocab": 97,
    "n_ctx": 2,
    "name": "big",
}

def SmallTransformer():
    return Transformer(CFG_SMALL)
