# Model Card: MobiMind-S1-v0 (Control Baseline)

- **Parameters**: 2,093 (FP32)
- **File Size**: 18.83 KB ONNX
- **Input**: 16-dimensional structured context feature vector
- **Outputs**:
  - Intent Logits (5 classes)
  - Action Logits (6 classes: `DoNothing`, `OpenApp`, `Tap`, `Back`, `Scroll`, `ToggleSetting`)
  - Risk Score (Sigmoid [0.0, 1.0])
  - Confidence Score (Sigmoid [0.0, 1.0])
- **Model Forward Latency (ARMv9-A CPU)**: P50: 1.30 us | P95: 1.35 us
