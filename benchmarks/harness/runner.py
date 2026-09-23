"""
MobiMind Benchmark Runner v0.1
Orchestrates hardware-level measurements on target Android device via ADB.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional

DEVICE_SERIAL = "e7c443b9"

def run_adb_command(cmd: str, serial: str = DEVICE_SERIAL) -> str:
    full_cmd = ["adb", "-s", serial, "shell"] + cmd.split()
    res = subprocess.run(full_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return ""
    return res.stdout.strip()

def get_battery_and_thermal(serial: str = DEVICE_SERIAL) -> Dict[str, Any]:
    raw_battery = run_adb_command("dumpsys battery", serial)
    level = None
    temp = None
    for line in raw_battery.splitlines():
        line = line.strip()
        if line.startswith("level:"):
            try:
                level = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("temperature:"):
            try:
                # Android reports temperature in tenths of a degree Celsius (e.g. 320 = 32.0C)
                temp = float(line.split(":")[1].strip()) / 10.0
            except ValueError:
                pass
    return {"battery_percent": level, "temp_celsius": temp}

def get_process_memory(package_name: str, serial: str = DEVICE_SERIAL) -> Dict[str, int]:
    raw_mem = run_adb_command(f"dumpsys meminfo {package_name}", serial)
    pss_kb = 0
    rss_kb = 0
    for line in raw_mem.splitlines():
        if "TOTAL PSS:" in line:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    pss_kb = int(parts[2])
                except ValueError:
                    pass
        elif "TOTAL RSS:" in line:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rss_kb = int(parts[2])
                except ValueError:
                    pass
    return {"pss_kb": pss_kb, "rss_kb": rss_kb}

def wait_for_thermal_baseline(target_temp: float = 35.0, timeout_sec: int = 180, serial: str = DEVICE_SERIAL) -> float:
    start_time = time.time()
    current_temp = get_battery_and_thermal(serial).get("temp_celsius", 40.0)
    print(f"Initial thermal state: {current_temp}°C (target: <= {target_temp}°C)")
    while current_temp and current_temp > target_temp and (time.time() - start_time) < timeout_sec:
        time.sleep(5)
        current_temp = get_battery_and_thermal(serial).get("temp_celsius", 40.0)
        print(f"Cooling down... Current: {current_temp}°C")
    return current_temp or 0.0

if __name__ == "__main__":
    print(f"Testing ADB connection with {DEVICE_SERIAL}...")
    state = get_battery_and_thermal()
    print(f"Device State: Battery: {state.get('battery_percent')}%, Temp: {state.get('temp_celsius')}°C")
