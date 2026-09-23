"""
Calibrated Parameter Configurations for v0.3
"""
from model_families import (
    StandardTransformer, ALBERTModel, MobileBERTModel, MobiMINDHybridModel,
    count_params
)

def build_calibrated_model(family: str, tier: str):
    if family == "standard_transformer":
        if tier == "10m":
            return StandardTransformer(vocab_size=8192, d_model=512, nhead=8, num_layers=3, dim_feedforward=1024) # ~10.5M
        elif tier == "25m":
            return StandardTransformer(vocab_size=8192, d_model=768, nhead=12, num_layers=4, dim_feedforward=1536) # ~24.8M
        elif tier == "50m":
            return StandardTransformer(vocab_size=8192, d_model=1024, nhead=16, num_layers=4, dim_feedforward=3072) # ~50.4M

    elif family == "albert":
        if tier == "10m":
            return ALBERTModel(vocab_size=8192, emb_dim=128, d_model=960, nhead=12, num_passes=6, dim_feedforward=3072) # ~10.2M
        elif tier == "25m":
            return ALBERTModel(vocab_size=8192, emb_dim=256, d_model=1408, nhead=16, num_passes=8, dim_feedforward=5120) # ~25.1M
        elif tier == "50m":
            return ALBERTModel(vocab_size=8192, emb_dim=256, d_model=1920, nhead=24, num_passes=12, dim_feedforward=7680) # ~48.2M

    elif family == "mobilebert":
        if tier == "10m":
            return MobileBERTModel(vocab_size=8192, d_model=576, bottleneck_dim=144, nhead=4, num_layers=4) # ~10.5M
        elif tier == "25m":
            return MobileBERTModel(vocab_size=8192, d_model=864, bottleneck_dim=216, nhead=8, num_layers=5) # ~24.6M
        elif tier == "50m":
            return MobileBERTModel(vocab_size=8192, d_model=1184, bottleneck_dim=296, nhead=8, num_layers=6) # ~50.0M

    elif family == "mobimind_hybrid":
        if tier == "10m":
            return MobiMINDHybridModel(vocab_size=8192, d_model=512, nhead=8, num_layers=3) # ~9.7M
        elif tier == "25m":
            return MobiMINDHybridModel(vocab_size=8192, d_model=800, nhead=16, num_layers=4) # ~24.8M
        elif tier == "50m":
            return MobiMINDHybridModel(vocab_size=8192, d_model=1152, nhead=16, num_layers=5) # ~50.2M

    raise ValueError(f"Unknown family {family} or tier {tier}")

if __name__ == "__main__":
    for tier in ["10m", "25m", "50m"]:
        print(f"\n[Calibrated Tier {tier.upper()}]")
        for fam in ["standard_transformer", "albert", "mobilebert", "mobimind_hybrid"]:
            m = build_calibrated_model(fam, tier)
            cnt = count_params(m)
            print(f"  {fam:<22} : {cnt:,} parameters ({cnt / 1e6:.2f}M)")
