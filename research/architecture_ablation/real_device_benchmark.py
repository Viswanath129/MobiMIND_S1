"""
MobiMIND Real Device Benchmark Runner
Connects to physical Snapdragon 6 Gen 4 via ADB, executes native NEON-optimized benchmark for each model candidate,
and parses verified physical measurements.
"""

import json
import subprocess
from typing import Dict, Any

MODEL_SPECS = {
    # arch, tier: (arch_name, tier_name, d_model, layers_or_passes, aux_dim)
    ("standard_transformer", "10m"): ("standard_transformer", "10m", 512, 3, 1024),
    ("albert", "10m"): ("albert", "10m", 960, 6, 3072),
    ("mobilebert", "10m"): ("mobilebert", "10m", 576, 4, 144),
    ("mobimind_hybrid", "10m"): ("mobimind_hybrid", "10m", 512, 3, 512),

    ("standard_transformer", "25m"): ("standard_transformer", "25m", 768, 4, 1536),
    ("albert", "25m"): ("albert", "25m", 1408, 8, 5120),
    ("mobilebert", "25m"): ("mobilebert", "25m", 864, 5, 216),
    ("mobimind_hybrid", "25m"): ("mobimind_hybrid", "25m", 800, 4, 800),

    ("standard_transformer", "50m"): ("standard_transformer", "50m", 1024, 4, 3072),
    ("albert", "50m"): ("albert", "50m", 1920, 12, 7680),
    ("mobilebert", "50m"): ("mobilebert", "50m", 1184, 6, 296),
    ("mobimind_hybrid", "50m"): ("mobimind_hybrid", "50m", 1152, 5, 1152),
}

def run_physical_model_bench(family: str, tier: str, device_serial: str = "e7c443b9") -> Dict[str, Any]:
    spec = MODEL_SPECS.get((family, tier))
    if not spec:
        raise ValueError(f"Unknown family {family} or tier {tier}")
    
    arch_name, tier_name, d_model, layers, aux = spec
    cmd = [
        "adb", "-s", device_serial, "shell",
        f"LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/physical_model_bench {arch_name} {tier_name} {d_model} {layers} {aux}"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out_json = res.stdout.strip()
    # Find start and end of JSON
    s_idx = out_json.find("{")
    e_idx = out_json.rfind("}")
    if s_idx == -1 or e_idx == -1:
        raise RuntimeError(f"Failed to parse JSON output: {out_json}")
    
    return json.loads(out_json[s_idx:e_idx+1])

if __name__ == "__main__":
    print("Testing device runner for mobimind_hybrid 10m...")
    data = run_physical_model_bench("mobimind_hybrid", "10m")
    print(json.dumps(data, indent=2))
