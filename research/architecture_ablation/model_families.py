"""
MobiMIND-S1 Architecture Families for v0.3 Capacity Scaling
Implements 4 architectures across 3 parameter tiers (~10M, ~25M, ~50M):
  A. Standard Transformer
  B. ALBERT-style (Factorized embedding + cross-layer weight sharing)
  C. MobileBERT-style (Intra-block linear bottlenecks)
  D. MobiMIND Hybrid (Multi-Head Attention + Gated Depthwise Conv1D FFN)

Output contract:
  - intent_logits (num_intents)
  - action_logits (num_actions)
  - risk (1, Sigmoid)
  - confidence (1, Sigmoid)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_INTENTS = 7
NUM_ACTIONS = 7

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size=8192, d_model=512, nhead=8, num_layers=3, dim_feedforward=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
        self.head_intent = nn.Linear(d_model, NUM_INTENTS)
        self.head_action = nn.Linear(d_model, NUM_ACTIONS)
        self.head_risk = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.head_conf = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        feat = self.encoder(x)
        cls_feat = self.norm(feat[:, 0, :])
        return (
            self.head_intent(cls_feat),
            self.head_action(cls_feat),
            self.head_risk(cls_feat),
            self.head_conf(cls_feat)
        )

class ALBERTModel(nn.Module):
    def __init__(self, vocab_size=8192, emb_dim=128, d_model=768, nhead=12, num_passes=6, dim_feedforward=2048):
        super().__init__()
        self.num_passes = num_passes
        # Factorized embedding: vocab -> emb_dim -> d_model
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.emb_project = nn.Linear(emb_dim, d_model)
        
        # Single shared Transformer layer reused across passes
        self.shared_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        
        self.head_intent = nn.Linear(d_model, NUM_INTENTS)
        self.head_action = nn.Linear(d_model, NUM_ACTIONS)
        self.head_risk = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.head_conf = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.emb_project(x)
        # Cross-layer parameter sharing (recursive pass)
        for _ in range(self.num_passes):
            x = self.shared_layer(x)
        cls_feat = self.norm(x[:, 0, :])
        return (
            self.head_intent(cls_feat),
            self.head_action(cls_feat),
            self.head_risk(cls_feat),
            self.head_conf(cls_feat)
        )

class MobileBERTBlock(nn.Module):
    def __init__(self, d_model, bottleneck_dim, nhead):
        super().__init__()
        self.input_project = nn.Linear(d_model, bottleneck_dim)
        self.attn = nn.MultiheadAttention(bottleneck_dim, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(bottleneck_dim)
        self.output_project = nn.Linear(bottleneck_dim, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Intra-block bottleneck
        b = self.input_project(x)
        attn_out, _ = self.attn(b, b, b)
        b = self.norm1(b + attn_out)
        h = self.norm2(x + self.output_project(b))
        out = self.norm3(h + self.ffn(h))
        return out

class MobileBERTModel(nn.Module):
    def __init__(self, vocab_size=8192, d_model=512, bottleneck_dim=128, nhead=4, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MobileBERTBlock(d_model, bottleneck_dim, nhead) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head_intent = nn.Linear(d_model, NUM_INTENTS)
        self.head_action = nn.Linear(d_model, NUM_ACTIONS)
        self.head_risk = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.head_conf = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        cls_feat = self.norm(x[:, 0, :])
        return (
            self.head_intent(cls_feat),
            self.head_action(cls_feat),
            self.head_risk(cls_feat),
            self.head_conf(cls_feat)
        )

class MobiMINDHybridBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        # Gated depthwise 1D conv FFN
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.gate = nn.Linear(d_model, d_model * 2)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # Conv + Gated MLP
        norm_x2 = self.norm2(x)
        conv_in = norm_x2.transpose(1, 2)
        conv_out = self.conv(conv_in).transpose(1, 2)
        
        gated = self.gate(conv_out)
        val, gate = gated.chunk(2, dim=-1)
        ffn_out = self.proj(val * F.silu(gate))
        return x + ffn_out

class MobiMINDHybridModel(nn.Module):
    def __init__(self, vocab_size=8192, d_model=512, nhead=8, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MobiMINDHybridBlock(d_model, nhead) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head_intent = nn.Linear(d_model, NUM_INTENTS)
        self.head_action = nn.Linear(d_model, NUM_ACTIONS)
        self.head_risk = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.head_conf = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        cls_feat = self.norm(x[:, 0, :])
        return (
            self.head_intent(cls_feat),
            self.head_action(cls_feat),
            self.head_risk(cls_feat),
            self.head_conf(cls_feat)
        )

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(family: str, tier: str):
    """
    Returns instantiated model configured for specified tier (~10M, ~25M, ~50M)
    """
    if family == "standard_transformer":
        if tier == "10m":
            return StandardTransformer(vocab_size=8192, d_model=512, nhead=8, num_layers=3, dim_feedforward=1024)
        elif tier == "25m":
            return StandardTransformer(vocab_size=8192, d_model=768, nhead=12, num_layers=4, dim_feedforward=2048)
        elif tier == "50m":
            return StandardTransformer(vocab_size=8192, d_model=1024, nhead=16, num_layers=4, dim_feedforward=3072)
            
    elif family == "albert":
        if tier == "10m":
            return ALBERTModel(vocab_size=8192, emb_dim=128, d_model=768, nhead=12, num_passes=6, dim_feedforward=2048)
        elif tier == "25m":
            return ALBERTModel(vocab_size=8192, emb_dim=256, d_model=1024, nhead=16, num_passes=8, dim_feedforward=4096)
        elif tier == "50m":
            return ALBERTModel(vocab_size=8192, emb_dim=256, d_model=1536, nhead=24, num_passes=12, dim_feedforward=6144)

    elif family == "mobilebert":
        if tier == "10m":
            return MobileBERTModel(vocab_size=8192, d_model=640, bottleneck_dim=160, nhead=4, num_layers=4)
        elif tier == "25m":
            return MobileBERTModel(vocab_size=8192, d_model=896, bottleneck_dim=224, nhead=8, num_layers=5)
        elif tier == "50m":
            return MobileBERTModel(vocab_size=8192, d_model=1152, bottleneck_dim=288, nhead=8, num_layers=6)

    elif family == "mobimind_hybrid":
        if tier == "10m":
            return MobiMINDHybridModel(vocab_size=8192, d_model=512, nhead=8, num_layers=3)
        elif tier == "25m":
            return MobiMINDHybridModel(vocab_size=8192, d_model=768, nhead=12, num_layers=4)
        elif tier == "50m":
            return MobiMINDHybridModel(vocab_size=8192, d_model=1024, nhead=16, num_layers=4)

    raise ValueError(f"Unknown family {family} or tier {tier}")

if __name__ == "__main__":
    print("--- Verifying Parameter Counts for All 12 Candidates ---")
    for tier in ["10m", "25m", "50m"]:
        print(f"\n[Tier {tier.upper()}]")
        for fam in ["standard_transformer", "albert", "mobilebert", "mobimind_hybrid"]:
            m = build_model(fam, tier)
            cnt = count_params(m)
            print(f"  {fam:<22} : {cnt:,} parameters ({cnt / 1e6:.2f}M)")
