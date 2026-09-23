# Benchmark Methodology

### Principles
1. **Physical Device Verification**: All benchmarks must execute on physical target hardware (Realme RMX5070 / Snapdragon 6 Gen 4).
2. **Separate Model vs. Agent Leaderboards**: Never conflate pure model forward pass with full mobile agent execution.
3. **Mathematical Invariant Assertions**: Every distribution must satisfy `P50 <= P90 <= P95 <= P99` and `Min <= P50 <= Max`.
4. **Thermal Baseline Gating**: Benchmark runs must record initial and final battery temperatures and discard runs where thermal throttling alters clock frequencies.
