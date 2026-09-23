"""
MobiMIND-S1 v0.3-R2 True Model-on-Device Benchmark Runner
Connects to physical Realme RMX5070 (Snapdragon 6 Gen 4), pushes actual trained ONNX models and data buffers,
executes native C++ onnxruntime runner (/system/lib64/libonnxruntime.so), and measures genuine model latency and RSS.
"""

import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List

BASE = Path("B:/projects/MobiMIND")
DEVICE_SERIAL = "e7c443b9"
REMOTE_DIR = "/data/local/tmp"

def run_actual_onnx_model_on_device(onnx_local_path: Path, num_iters: int = 15) -> Dict[str, Any]:
    """
    Pushes actual ONNX file and external .data file to device,
    executes real_onnx_device_runner, and retrieves verified physical metrics.
    """
    if not onnx_local_path.exists():
        raise FileNotFoundError(f"Local ONNX model not found: {onnx_local_path}")
    
    data_local_path = Path(str(onnx_local_path) + ".data")
    
    remote_onnx = f"{REMOTE_DIR}/{onnx_local_path.name}"
    remote_data = f"{REMOTE_DIR}/{data_local_path.name}"
    
    print(f"       Pushing actual ONNX graph to device: {onnx_local_path.name} ...", flush=True)
    subprocess.run(["adb", "-s", DEVICE_SERIAL, "push", str(onnx_local_path), remote_onnx], check=True, capture_output=True)
    
    if data_local_path.exists():
        size_mb = data_local_path.stat().st_size / (1024 * 1024)
        print(f"       Pushing actual parameter weights buffer ({size_mb:.1f} MB) to device ...", flush=True)
        subprocess.run(["adb", "-s", DEVICE_SERIAL, "push", str(data_local_path), remote_data], check=True, capture_output=True)
        
    cmd = [
        "adb", "-s", DEVICE_SERIAL, "shell",
        f"LD_LIBRARY_PATH=/system/lib64 {REMOTE_DIR}/real_onnx_device_runner {remote_onnx} {num_iters}"
    ]
    
    print(f"       Executing actual model graph on physical Snapdragon 6 Gen 4 CPU...", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Device benchmark failed ({res.returncode}): {res.stderr}\n{res.stdout}")
        
    out_str = res.stdout.strip()
    s_idx = out_str.find("{")
    e_idx = out_str.rfind("}")
    if s_idx == -1 or e_idx == -1:
        raise RuntimeError(f"Failed to parse JSON from device output: {out_str}")
        
    bench_data = json.loads(out_str[s_idx:e_idx+1])
    
    # Clean up large model files from device tmp to prevent filling RAM disk
    subprocess.run(["adb", "-s", DEVICE_SERIAL, "shell", f"rm -f {remote_onnx} {remote_data}"], check=True, capture_output=True)
    
    return bench_data

if __name__ == "__main__":
    test_model = BASE / "benchmarks/runs/v0.3-r1/onnx/mobimind_hybrid_10m.onnx"
    print("Testing on-device execution for actual model:", test_model.name)
    metrics = run_actual_onnx_model_on_device(test_model, num_iters=10)
    print(json.dumps(metrics, indent=2))
