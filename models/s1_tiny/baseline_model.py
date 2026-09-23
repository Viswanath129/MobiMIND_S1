"""
MobiMind-S1-v0: Deliberately Tiny Baseline System-1 Encoder.
Takes structured mobile context features and predicts:
  1. Intent (classification)
  2. Action (classification: OpenApp, Tap, Back, Scroll, ToggleSetting, DoNothing)
  3. Risk (sigmoid regression [0.0, 1.0])
  4. Confidence (sigmoid regression [0.0, 1.0])

Exports to ONNX for cross-backend benchmarking (CPU / GPU / Hexagon).
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

ACTIONS = [
    "DoNothing",
    "OpenApp",
    "Tap",
    "Back",
    "Scroll",
    "ToggleSetting"
]

INTENTS = [
    "no_op",
    "system_control",
    "connectivity_control",
    "power_management",
    "display_control"
]

class MobiMindS1Tiny(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.head_intent = nn.Linear(hidden_dim, len(INTENTS))
        self.head_action = nn.Linear(hidden_dim, len(ACTIONS))
        self.head_risk = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.head_confidence = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        intent_logits = self.head_intent(features)
        action_logits = self.head_action(features)
        risk = self.head_risk(features)
        confidence = self.head_confidence(features)
        return intent_logits, action_logits, risk, confidence

def encode_context(context_dict: dict) -> torch.Tensor:
    # Feature vector layout (16-dim):
    # [battery_normalized, net_wifi, net_cell, net_none, net_airplane,
    #  app_hash_sin, app_hash_cos, screen_hash_sin, screen_hash_cos,
    #  event_hash_sin, event_hash_cos, goal_hash_sin, goal_hash_cos,
    #  0, 0, 0]
    vec = [0.0] * 16
    vec[0] = context_dict.get("battery", 50) / 100.0
    net = context_dict.get("network", "wifi")
    vec[1] = 1.0 if net == "wifi" else 0.0
    vec[2] = 1.0 if net == "cellular" else 0.0
    vec[3] = 1.0 if net == "none" else 0.0
    vec[4] = 1.0 if net == "airplane" else 0.0

    def hash_enc(s: str):
        h = abs(hash(s)) % 1000
        return np.sin(h), np.cos(h)

    s1, c1 = hash_enc(context_dict.get("app", ""))
    vec[5], vec[6] = float(s1), float(c1)
    s2, c2 = hash_enc(context_dict.get("screen", ""))
    vec[7], vec[8] = float(s2), float(c2)
    s3, c3 = hash_enc(context_dict.get("event", ""))
    vec[9], vec[10] = float(s3), float(c3)
    s4, c4 = hash_enc(context_dict.get("goal", ""))
    vec[11], vec[12] = float(s4), float(c4)
    return torch.tensor([vec], dtype=torch.float32)

def train_and_export():
    workspace = Path(__file__).resolve().parent
    dataset_path = workspace / "sample_contexts.json"
    with open(dataset_path, "r") as f:
        data = json.load(f)

    model = MobiMindS1Tiny(input_dim=16, hidden_dim=32)
    model.train()
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MobiMind-S1-v0 Total Parameters: {total_params}")

    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train for 150 epochs on sample dataset
    for epoch in range(150):
        total_loss = 0.0
        for item in data:
            x = encode_context(item["context"])
            pred = item["prediction"]
            intent_idx = torch.tensor([INTENTS.index(pred["intent"]) if pred["intent"] in INTENTS else 0])
            action_idx = torch.tensor([ACTIONS.index(pred["action"]) if pred["action"] in ACTIONS else 0])
            risk_tgt = torch.tensor([[pred["risk"]]], dtype=torch.float32)
            conf_tgt = torch.tensor([[pred["confidence"]]], dtype=torch.float32)

            optimizer.zero_grad()
            i_logits, a_logits, r_val, c_val = model(x)

            loss = (
                criterion_ce(i_logits, intent_idx)
                + criterion_ce(a_logits, action_idx)
                + criterion_mse(r_val, risk_tgt)
                + criterion_mse(c_val, conf_tgt)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    print(f"Training completed. Final batch loss: {total_loss:.4f}")

    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 16, dtype=torch.float32)
    onnx_path = workspace / "mobimind_s1_v0.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["context_features"],
        output_names=["intent_logits", "action_logits", "risk", "confidence"],
        dynamic_axes={"context_features": {0: "batch_size"}},
        opset_version=14
    )
    print(f"Exported ONNX model to: {onnx_path}")
    print(f"ONNX file size: {onnx_path.stat().st_size} bytes ({onnx_path.stat().st_size / 1024:.2f} KB)")

    # Verify ONNX model with onnxruntime
    ort_session = ort.InferenceSession(str(onnx_path))
    test_x = encode_context(data[0]["context"]).numpy()
    outputs = ort_session.run(None, {"context_features": test_x})
    pred_intent = INTENTS[np.argmax(outputs[0])]
    pred_action = ACTIONS[np.argmax(outputs[1])]
    risk = float(outputs[2][0][0])
    confidence = float(outputs[3][0][0])

    print("\n--- Model Verification on Sample 0 (Wi-Fi off) ---")
    print(f"Expected: Intent=system_control, Action=ToggleSetting")
    print(f"Predicted: Intent={pred_intent}, Action={pred_action}, Risk={risk:.2f}, Confidence={confidence:.2f}")

if __name__ == "__main__":
    train_and_export()
