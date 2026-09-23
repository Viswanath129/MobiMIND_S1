# System Architecture

The MobiMIND-S1 system operates on an event-driven, hierarchical cognitive loop:

1. **Event Reception**: Sensors, notifications, user utterances, UI state changes.
2. **Context Delta Engine**: Compares incoming signals against the cached phone context.
3. **Hierarchy of Agency**:
   - **Level 0 (Deterministic)**: Hard rules bypass neural inference (0 ms model compute).
   - **Level 1 (Heuristic)**: Ultra-fast classifiers (<0.01 ms).
   - **Level 2 (MobiMind-S1)**: 2K-100M parameter semantic decision model (0.26 ms).
   - **Level 3 (Local Reasoning)**: 0.5B-4B LLM for ambiguous or multi-step tasks.
   - **Level 4 (Cloud)**: Remote reasoning escalation.
4. **Action & Capability Resolver**: Selects the optimal execution path:
   - `DIRECT_API` (In-process Binder / ContentResolver)
   - `EVENT` (ContentObserver / AccessibilityEvent)
   - `UI_NODE` (AccessibilityNodeInfo tree traversal)
5. **Verification Strategy Engine**: Verifies state transitions using event-driven listeners instead of fixed polling timeouts.
