"""
Fusion-model(ultra efficient)

Multi grouped query attention + feedforward + Locality-Sensitive Hashing

"""

import torch
from torch import Tensor, nn
from zeta.nn import FeedForward, OutputHead
from typing import Optional
import numpy as np

class MultiQueryAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, -1).transpose(1, 2), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * (self.dim ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class LocalitySensitiveHashing(nn.Module):
    def __init__(self, input_dim: int, hash_dim: int, num_hashes: int):
        super().__init__()
        self.hash_projections = nn.Parameter(torch.randn(num_hashes, input_dim, hash_dim))

    def forward(self, x: Tensor) -> Tensor:
        return torch.sign(torch.einsum('bnd,hde->bhne', x, self.hash_projections))

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mult: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiQueryAttention(dim, heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dim, mult, swish=True, post_act_ln=True, dropout=dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        attn_out = self.attn(self.ln1(x))
        if mask is not None:
            attn_out = attn_out * mask.unsqueeze(-1)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x

class Fusion(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mult: int,
        depth: int,
        num_tokens: int,
        max_seq_len: int,
        dropout: float = 0.1,
        lsh_dim: int = 16,
        num_hashes: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        self.embed = nn.Embedding(num_tokens, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, dim))
        
        self.lsh = LocalitySensitiveHashing(dim, lsh_dim, num_hashes)
        
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, mult, dropout)
            for _ in range(depth)
        ])
        
        self.ln_out = nn.LayerNorm(dim)
        self.output_head = OutputHead(dim, num_tokens)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        b, n = x.shape
        assert n <= self.max_seq_len, f"Input sequence length {n} exceeds maximum sequence length {self.max_seq_len}"

        x = self.embed(x) + self.pos_embed[:, :n]
        
        lsh_out = self.lsh(x)
        
        if mask is not None:
            mask = mask[:, :n, :n]
        
        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_out(x)
        x = self.output_head(x)
        
        return x, lsh_out

def create_causal_mask(seq_len: int) -> Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.float().masked_fill(mask, 0.0).masked_fill(~mask, 1.0)

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    dim = 512
    heads = 8
    mult = 4
    depth = 8
    num_tokens = 100
    max_seq_len = 100
    batch_size = 32

    model = Fusion(
        dim=dim,
        heads=heads,
        mult=mult,
        depth=depth,
        num_tokens=num_tokens,
        max_seq_len=max_seq_len
    )
    
    x = torch.randint(0, num_tokens, (batch_size, max_seq_len))
    mask = create_causal_mask(max_seq_len).unsqueeze(0).repeat(batch_size, 1, 1)

    with torch.no_grad():
        y, lsh_out = model(x, mask)

    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output shape: {y.shape}")
    print(f"LSH output shape: {lsh_out.shape}")

    if __name__ == "__main__":
     main()
     