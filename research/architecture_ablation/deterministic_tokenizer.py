"""
Deterministic Tokenizer for MobiMIND-S1
Guarantees identical token IDs across Python processes, platforms, and runtimes without relying on Python built-in hash() randomization.
"""

import hashlib
import json
import torch
from typing import Dict, Any, List

TOKENIZER_VERSION = "1.0.0"
VOCAB_SIZE = 8192
MAX_SEQ_LEN = 16

# Special Token IDs
PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 1
SEP_TOKEN_ID = 2
UNK_TOKEN_ID = 3

def deterministic_hash(s: str, max_val: int = 7000, offset: int = 100) -> int:
    """Computes a stable integer from string using SHA-256."""
    if not s:
        return PAD_TOKEN_ID
    digest = hashlib.sha256(s.encode("utf-8")).hexdigest()
    # Take first 8 bytes
    val = int(digest[:8], 16)
    return (val % max_val) + offset

def tokenize_context(ctx: Dict[str, Any], seq_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    """
    Deterministically tokenizes a mobile context dictionary into a fixed-length tensor.
    Slots:
      0: [CLS]
      1: app token
      2: screen token
      3: battery token (normalized to 100-200)
      4: network token
      5: event token
      6..15: goal tokens (word tokens)
    """
    tokens = [CLS_TOKEN_ID]
    
    # Slot 1: App
    app_str = str(ctx.get("app", "")).lower().strip()
    tokens.append(deterministic_hash(app_str))
    
    # Slot 2: Screen
    screen_str = str(ctx.get("screen", "")).lower().strip()
    tokens.append(deterministic_hash(screen_str))
    
    # Slot 3: Battery (100 + clamped percentage)
    bat = ctx.get("battery", 50)
    try:
        bat_val = int(bat)
    except (ValueError, TypeError):
        bat_val = 50
    tokens.append(min(200, max(100, bat_val + 100)))
    
    # Slot 4: Network
    net_str = str(ctx.get("network", "")).lower().strip()
    tokens.append(deterministic_hash(net_str))
    
    # Slot 5: Event
    event_str = str(ctx.get("event", "")).lower().strip()
    tokens.append(deterministic_hash(event_str))
    
    # Slot 6..seq_len: Goal
    goal = str(ctx.get("goal", "")).lower().strip()
    words = goal.split()
    for w in words:
        if len(tokens) >= seq_len:
            break
        tokens.append(deterministic_hash(w))
        
    while len(tokens) < seq_len:
        tokens.append(PAD_TOKEN_ID)
        
    return torch.tensor([tokens[:seq_len]], dtype=torch.long)

def get_tokenizer_hash() -> str:
    """Returns SHA256 of tokenizer source code for provenance tracking."""
    import inspect
    src = inspect.getsource(tokenize_context)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()

if __name__ == "__main__":
    test_ctx = {"app": "settings", "screen": "wifi", "battery": 28, "network": "wifi", "event": "user_request", "goal": "turn off wifi"}
    t1 = tokenize_context(test_ctx)
    t2 = tokenize_context(test_ctx)
    assert torch.equal(t1, t2), "Deterministic check failed!"
    print(f"Deterministic Tokenizer v{TOKENIZER_VERSION} OK. Hash: {get_tokenizer_hash()[:16]}")
    print("Tokens:", t1.tolist())
