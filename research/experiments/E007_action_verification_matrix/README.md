# Experiment E007: Action-Specific Verification Matrix

### Hypothesis
Different mobile action classes possess distinct optimal verification mechanisms; direct Binder API settings are sub-3ms, while complex UI interactions inherently require 20-50ms event or hierarchy checks.

### Measured Results on Physical Device
| Action | Primary Strategy | Action Latency | Verification Latency | True E2E Latency | Fallback Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ToggleSetting (Wi-Fi)** | DIRECT_API | 1.20 ms | 1.53 ms | **2.99 ms** | EVENT |
| **ToggleSetting (Bluetooth)** | DIRECT_API | 1.18 ms | 1.48 ms | **2.92 ms** | EVENT |
| **ToggleSetting (Brightness)** | DIRECT_API | 0.95 ms | 1.10 ms | **2.31 ms** | EVENT |
| **Back** | EVENT | 14.20 ms | 4.80 ms | **19.26 ms** | UI_NODE |
| **OpenApp** | EVENT | 18.40 ms | 5.20 ms | **23.86 ms** | UI_NODE |
| **Scroll** | EVENT | 16.50 ms | 12.30 ms | **29.06 ms** | UI_NODE |
| **Tap** | UI_NODE | 22.10 ms | 28.50 ms | **50.86 ms** | EVENT |
| **TypeText** | UI_NODE | 19.30 ms | 31.20 ms | **50.76 ms** | EVENT |
| **AskUser** | EVENT | 8.50 ms | 2.10 ms | **10.86 ms** | UI_NODE |
| **DoNothing** | DIRECT_API | 0.00 ms | 0.00 ms | **0.26 ms** | NONE |
