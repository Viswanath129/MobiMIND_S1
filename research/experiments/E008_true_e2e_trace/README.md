# Experiment E008: True E2E Latency Trace

### Complete Pipeline Decomposition (Microseconds & Milliseconds)
```text
T_total (Direct API) =
    T_context   (0.050 ms / 50.0 us)
  + T_delta     (0.015 ms / 15.0 us)
  + T_model     (0.0013 ms / 1.3 us)
  + T_policy    (0.010 ms / 10.0 us)
  + T_dispatch  (0.120 ms / 120.0 us)
  + T_android   (1.080 ms / 1,080.0 us)
  + T_verify    (1.530 ms / 1,530.0 us)
  ─────────────────────────────────────
  True E2E =    2.8063 ms (~2.81 ms)
```
